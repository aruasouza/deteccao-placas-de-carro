import torch
import torch.nn as nn
from pathlib import Path

IMG_W, IMG_H = 100, 32
NUM_CHARS    = 7

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvBNReLU(ch, ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + x)

class PlateOCRNet(nn.Module):
    """
    CNN multi-head para reconhecimento de 7 caracteres de placa.

    Cada head i tem num_classes_per_pos[i] saídas:
      - 26 para posições de letra
      - 10 para posições de dígito

    Isso elimina predições estruturalmente inválidas (ex: letra em posição de dígito)
    e reduz o espaço de busca de cada head.

    Entrada:  (B, 1, 32, 100)
    Saída:    lista de 7 tensores (B, num_classes_per_pos[i])
    """

    def __init__(self, num_classes_per_pos: list[int], dropout: float = 0.3):
        super().__init__()
        assert len(num_classes_per_pos) == NUM_CHARS
        self.num_classes_per_pos = num_classes_per_pos

        self.backbone = nn.Sequential(
            ConvBNReLU(1, 32),
            ConvBNReLU(32, 32),
            nn.MaxPool2d(2, 2),                                          # → (B,32,16,50)

            ConvBNReLU(32, 64),
            ResidualBlock(64),
            nn.MaxPool2d(2, 2),                                          # → (B,64,8,25)

            ConvBNReLU(64, 128),
            ResidualBlock(128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 1)),  # → (B,128,4,13)

            ConvBNReLU(128, 256),
            ResidualBlock(256),
            nn.MaxPool2d(2, 2),                                          # → (B,256,2,6)

            ConvBNReLU(256, 256),
            nn.AdaptiveAvgPool2d((1, NUM_CHARS)),                        # → (B,256,1,7)
        )

        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout / 2),
                nn.Linear(128, num_classes_per_pos[i]),
            )
            for i in range(NUM_CHARS)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feat = self.backbone(x).squeeze(2)   # (B, 256, 7)
        feat = self.dropout(feat)
        return [self.heads[i](feat[:, :, i]) for i in range(NUM_CHARS)]
    
def export_onnx(model: PlateOCRNet, path: Path, device):
    model.eval()
 
    max_classes = max(model.num_classes_per_pos)
 
    class ONNXWrapper(nn.Module):
        def __init__(self, m, max_cls):
            super().__init__()
            self.m       = m
            self.max_cls = max_cls
 
        def forward(self, x):
            logits = self.m(x)           # lista de 7 tensores (B, n_classes_i)
            B      = x.size(0)
            padded = []
            for lg in logits:
                if lg.size(1) < self.max_cls:
                    pad = torch.full((B, self.max_cls - lg.size(1)),
                                     float('-inf'), device=x.device)
                    lg = torch.cat([lg, pad], dim=1)
                padded.append(lg)
            return torch.stack(padded, dim=1)   # (B, 7, max_classes)
 
    wrapper = ONNXWrapper(model, max_classes).to(device)
    dummy   = torch.zeros(1, 1, IMG_H, IMG_W, device=device)
 
    torch.onnx.export(
        wrapper, dummy, str(path),
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes=None,
        opset_version=18,
        do_constant_folding=True,
        dynamo=True,
        export_params=True
    )
    print(f"[ONNX] Modelo exportado → {path}")
    
LETRAS  = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'   # índices 0-25
DIGITOS = '0123456789'                    # índices 0-9

# Máscara de posições: True = letra, False = dígito
LAYOUT_MASK: dict[str, list[bool]] = {
    'BR': [True,  True,  True,  False, False, False, False],
    'ME': [True,  True,  True,  False, True,  False, False],
}

def vocab_for_pos(layout: str, pos: int) -> str:
    """Retorna o vocabulário válido para a posição dada."""
    return LETRAS if LAYOUT_MASK[layout][pos] else DIGITOS
    
def classes_per_pos(layout: str) -> list[int]:
    """Retorna lista com número de classes por posição."""
    return [len(vocab_for_pos(layout, i)) for i in range(7)]

if __name__ == '__main__':
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    export_dir = Path('onnx_models')
    for layout in ('BR','ME'):
        best_ckpt_path = f'checkpoints/{layout}/best.pth'
        n_classes = classes_per_pos(layout)
        model    = PlateOCRNet(n_classes, dropout=0.3).to(device)
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        onnx_path = export_dir / f'plate_ocr_{layout.lower()}.onnx'
        export_onnx(model, onnx_path, device)
        print(f'Exportado para {onnx_path}')