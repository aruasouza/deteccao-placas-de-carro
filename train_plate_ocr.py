"""
Treinamento de rede neural para reconhecimento de caracteres de placas veiculares.

Arquitetura: CNN + classificadores paralelos (um por posição)
Entrada: imagem em escala de cinza 100x32 (placa recortada e retificada)
Saída: 7 caracteres com número de classes ajustado por posição:
    BR: L  L  L  N  N  N  N   → 26 26 26 10 10 10 10
    ME: L  L  L  N  L  N  N   → 26 26 26 10 26 10 10

Pré-requisito:
    python prepare_dataset.py   # roda YOLO + warp nas imagens brutas e salva as placas

Uso:
    python train_plate_ocr.py --layout BR [--epochs 100] [--batch 64] [--lr 3e-3]
    python train_plate_ocr.py --layout ME

Se checkpoints/<layout>/best.pth existir, retoma automaticamente.
Para forçar início do zero: --no-resume
"""

import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms as T

# ─────────────────────────── Constantes ───────────────────────────────────────

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

IMG_W, IMG_H = 100, 32
NUM_CHARS    = 7

DATASETS_ROOT = Path('datasets/dataset1')
TRAIN_IMGS    = DATASETS_ROOT / 'placas' / 'train'   # placas recortadas e retificadas
VAL_IMGS      = DATASETS_ROOT / 'placas' / 'val'     # geradas por prepare_dataset.py
ANNOT_BR      = DATASETS_ROOT / 'bruto' / 'cars-br'
ANNOT_ME      = DATASETS_ROOT / 'bruto' / 'cars-me'

def layout_dirs(layout: str) -> tuple[Path, Path, Path]:
    ckpt  = Path('checkpoints') / layout
    plots = Path('plots')       / layout
    exp   = Path('onnx_models')
    for d in (ckpt, plots, exp):
        d.mkdir(parents=True, exist_ok=True)
    return ckpt, plots, exp

# ─────────────────────────── Dataset ──────────────────────────────────────────

@dataclass
class PlateAnnotation:
    stem:   str
    plate:  str
    layout: str   # "Brazilian" | "Mercosul"


def parse_annotation(txt_path: Path) -> Optional[PlateAnnotation]:
    data = {}
    try:
        for line in txt_path.read_text(encoding='utf-8', errors='ignore').splitlines():
            if ':' in line:
                k, v = line.split(':', 1)
                data[k.strip().lower()] = v.strip()
    except Exception:
        return None
    plate  = data.get('plate', '').upper().replace('-', '').replace(' ', '')
    layout = data.get('layout', '')
    if len(plate) != 7:
        return None
    return PlateAnnotation(stem=txt_path.stem, plate=plate, layout=layout)


def build_annotation_index(annot_dirs: list[Path]) -> dict[str, PlateAnnotation]:
    index: dict[str, PlateAnnotation] = {}
    for annot_dir in annot_dirs:
        for txt_path in annot_dir.glob('*.txt'):
            ann = parse_annotation(txt_path)
            if ann is not None:
                index[ann.stem] = ann
    return index


def normalize_layout(layout_field: str) -> Optional[str]:
    lf = layout_field.lower()
    if 'brazil' in lf:
        return 'BR'
    if 'mercosul' in lf or 'mercosur' in lf:
        return 'ME'
    return None


def encode_plate(plate: str, layout: str) -> Optional[torch.Tensor]:
    """
    Converte string de placa em tensor de índices [NUM_CHARS].
    Cada índice é relativo ao vocabulário da posição (0-25 para letra, 0-9 para dígito).
    """
    indices = []
    for pos, ch in enumerate(plate[:NUM_CHARS]):
        vocab = vocab_for_pos(layout, pos)
        if ch not in vocab:
            return None
        indices.append(vocab.index(ch))
    return torch.tensor(indices, dtype=torch.long)


class PlateDataset(Dataset):
    def __init__(self, img_dir: Path, annot_index: dict[str, PlateAnnotation],
                 target_layout: str, augment: bool = False):
        self.augment       = augment
        self.target_layout = target_layout
        self.samples: list[tuple[Path, torch.Tensor]] = []

        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in img_extensions:
                continue
            ann = annot_index.get(img_path.stem)
            if ann is None:
                continue
            if normalize_layout(ann.layout) != target_layout:
                continue
            label = encode_plate(ann.plate, target_layout)
            if label is None:
                continue
            self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Nenhuma amostra '{target_layout}' encontrada em '{img_dir}'. "
                "Verifique os caminhos e anotações."
            )

        self.aug_transform = T.Compose([
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4)],     p=0.5),
        ])

    def __len__(self):
        return len(self.samples)

    def _load_gray(self, img_path: Path) -> np.ndarray:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Não foi possível carregar: {img_path}")
        return cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img    = self._load_gray(img_path).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).unsqueeze(0)

        if self.augment:
            if torch.rand(1).item() < 0.4:
                tensor = (tensor + torch.randn_like(tensor) * 0.05).clamp(0.0, 1.0)
            if torch.rand(1).item() < 0.3:
                tensor = T.RandomPerspective(distortion_scale=0.15, p=1.0)(tensor)
            tensor = self.aug_transform(tensor)

        return tensor, label


# ─────────────────────────── Modelo ───────────────────────────────────────────

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


# ─────────────────────────── Métricas ─────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.sum   += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0


def compute_accuracy(logits_list: list[torch.Tensor],
                     targets: torch.Tensor) -> tuple[float, float]:
    B = targets.size(0)
    correct_chars = correct_plates = 0
    for b in range(B):
        all_ok = True
        for i, lg in enumerate(logits_list):
            if lg[b].argmax().item() == targets[b, i].item():
                correct_chars += 1
            else:
                all_ok = False
        if all_ok:
            correct_plates += 1
    return correct_chars / (B * NUM_CHARS), correct_plates / B


# ─────────────────────────── Treinamento ──────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device,
                    scaler=None) -> dict:
    model.train()
    loss_m, char_m, plate_m = AverageMeter(), AverageMeter(), AverageMeter()

    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = sum(criterion(logits[i], targets[:, i]) for i in range(NUM_CHARS))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = sum(criterion(logits[i], targets[:, i]) for i in range(NUM_CHARS))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scheduler.step()

        B = imgs.size(0)
        char_acc, plate_acc = compute_accuracy(logits, targets)
        loss_m.update(loss.item() / NUM_CHARS, B)
        char_m.update(char_acc, B)
        plate_m.update(plate_acc, B)

    return {'loss': loss_m.avg, 'char_acc': char_m.avg, 'plate_acc': plate_m.avg}


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    loss_m, char_m, plate_m = AverageMeter(), AverageMeter(), AverageMeter()

    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        logits = model(imgs)
        loss = sum(criterion(logits[i], targets[:, i]) for i in range(NUM_CHARS))
        B = imgs.size(0)
        char_acc, plate_acc = compute_accuracy(logits, targets)
        loss_m.update(loss.item() / NUM_CHARS, B)
        char_m.update(char_acc, B)
        plate_m.update(plate_acc, B)

    return {'loss': loss_m.avg, 'char_acc': char_m.avg, 'plate_acc': plate_m.avg}


# ─────────────────────────── Gráficos ─────────────────────────────────────────

# Cores hexadecimais — compatíveis com qualquer versão do matplotlib
_COLORS = {
    'train_loss':      '#d62728',   # vermelho
    'val_loss':        '#ff7f0e',   # laranja
    'train_char_acc':  '#1f77b4',   # azul
    'val_char_acc':    '#17becf',   # ciano
    'train_plate_acc': '#2ca02c',   # verde
    'val_plate_acc':   '#98df8a',   # verde claro
    'lr':              '#9467bd',   # roxo
}


def save_training_plots(history: dict, epoch: int, layout: str, save_dir: Path):
    epochs = list(range(1, epoch + 2))

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(f'Evolução do Treinamento — Plate OCR ({layout})',
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    metrics = [
        ('loss',      'Loss (cross-entropy média)'),
        ('char_acc',  'Acurácia por Caractere'),
        ('plate_acc', 'Acurácia por Placa (exata)'),
    ]

    for col, (key, title) in enumerate(metrics):
        ax = fig.add_subplot(gs[col])
        ax.plot(epochs, history[f'train_{key}'],
                color=_COLORS[f'train_{key}'], label='Treino', linewidth=2)
        ax.plot(epochs, history[f'val_{key}'],
                color=_COLORS[f'val_{key}'], label='Val', linewidth=2, linestyle='--')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Época')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if 'acc' in key:
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    if history.get('lr'):
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(history['lr'], color=_COLORS['lr'], linewidth=1)
        ax2.set_title('Learning Rate Schedule')
        ax2.set_xlabel('Iteração')
        ax2.set_ylabel('LR')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'lr_schedule.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)


def save_sample_predictions(model, loader, device, layout: str,
                             save_dir: Path, n: int = 16):
    model.eval()
    imgs_all, targets_all, preds_all = [], [], []

    with torch.no_grad():
        for imgs, targets in loader:
            logits = model(imgs.to(device))
            pred_idx = torch.stack([lg.argmax(1) for lg in logits], dim=1).cpu()
            imgs_all.append(imgs.cpu())
            targets_all.append(targets)
            preds_all.append(pred_idx)
            if sum(len(t) for t in targets_all) >= n:
                break

    imgs_all    = torch.cat(imgs_all)[:n]
    targets_all = torch.cat(targets_all)[:n]
    preds_all   = torch.cat(preds_all)[:n]

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 1.8))
    axes = axes.flatten()

    for i in range(n):
        gt = ''.join(vocab_for_pos(layout, c)[targets_all[i, c].item()]
                     for c in range(NUM_CHARS))
        pr = ''.join(vocab_for_pos(layout, c)[preds_all[i, c].item()]
                     for c in range(NUM_CHARS))
        color = 'green' if gt == pr else 'red'
        axes[i].imshow(imgs_all[i, 0].numpy(), cmap='gray', aspect='auto')
        axes[i].set_title(f'GT: {gt}\nPred: {pr}', fontsize=9, color=color)
        axes[i].axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Amostras de Validação ({layout})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'sample_predictions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────── Export ONNX ──────────────────────────────────────

def export_onnx(model: PlateOCRNet, path: Path, device):
    model.eval()

    class ONNXWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            logits = self.m(x)
            return torch.stack([lg.argmax(dim=1) for lg in logits], dim=1)

    wrapper = ONNXWrapper(model).to(device)
    dummy   = torch.zeros(1, 1, IMG_H, IMG_W, device=device)

    torch.onnx.export(
        wrapper, dummy, str(path),
        input_names=['input'],
        output_names=['char_indices'],
        dynamic_axes={'input': {0: 'batch'}, 'char_indices': {0: 'batch'}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[ONNX] Modelo exportado → {path}")


# ─────────────────────────── Main ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Treinar OCR de placas veiculares')
    p.add_argument('--layout',    type=str,   required=True, choices=['BR', 'ME'],
                   help='Tipo de placa a treinar')
    p.add_argument('--epochs',    type=int,   default=100)
    p.add_argument('--batch',     type=int,   default=64)
    p.add_argument('--lr',        type=float, default=3e-3)
    p.add_argument('--workers',   type=int,   default=4)
    p.add_argument('--dropout',   type=float, default=0.3)
    p.add_argument('--patience',  type=int,   default=20,
                   help='Early stopping: épocas sem melhora em val plate_acc')
    p.add_argument('--no-resume', action='store_true',
                   help='Força início do zero, ignorando checkpoint existente')
    return p.parse_args()


def main():
    args   = parse_args()
    layout = args.layout

    ckpt_dir, plots_dir, export_dir = layout_dirs(layout)

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"[Config]  Layout: {layout}  |  Device: {device}  |  AMP: {use_amp}")

    n_classes = classes_per_pos(layout)
    print(f"[Modelo]  Classes por posição: {n_classes}")

    # ── Anotações ──────────────────────────────────────────────────────────────
    print("[Dataset] Indexando anotações...")
    annot_index = build_annotation_index([ANNOT_BR, ANNOT_ME])
    print(f"          {len(annot_index)} anotações encontradas.")

    train_ds = PlateDataset(TRAIN_IMGS, annot_index, layout, augment=True)
    val_ds   = PlateDataset(VAL_IMGS,   annot_index, layout, augment=False)
    print(f"[Dataset] Treino: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # ── Modelo ─────────────────────────────────────────────────────────────────
    model    = PlateOCRNet(n_classes, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Modelo]  Parâmetros treináveis: {n_params:,}")

    criterion   = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler   = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps,
                             pct_start=0.1, div_factor=10, final_div_factor=100)
    scaler      = torch.amp.GradScaler() if use_amp else None

    start_epoch    = 0
    best_plate_acc = 0.0
    no_improve     = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_char_acc': [], 'val_char_acc': [],
        'train_plate_acc': [], 'val_plate_acc': [],
        'lr': [],
    }

    # ── Resume automático ──────────────────────────────────────────────────────
    best_ckpt_path = ckpt_dir / 'best.pth'
    if not args.no_resume and best_ckpt_path.exists():
        print(f"[Resume]  Checkpoint encontrado: {best_ckpt_path}")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch    = ckpt['epoch'] + 1
        best_plate_acc = ckpt.get('best_plate_acc', 0.0)
        history        = ckpt.get('history', history)
        print(f"          Retomando da época {start_epoch}  "
              f"(melhor plate_acc: {best_plate_acc:.2%})")
    else:
        print("[Resume]  Iniciando do zero.")

    # ── Loop de treinamento ────────────────────────────────────────────────────
    print(f"\n{'Época':>6} | {'Loss Tr':>9} {'Loss Va':>9} | "
          f"{'Char Tr':>9} {'Char Va':>9} | "
          f"{'Placa Tr':>9} {'Placa Va':>9} | {'Tempo':>6}")
    print('-' * 90)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        history['lr'].extend([scheduler.get_last_lr()[0]] * len(train_loader))

        tr = train_one_epoch(model, train_loader, criterion, optimizer,
                             scheduler, device, scaler)
        va = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        history['train_loss'].append(tr['loss'])
        history['val_loss'].append(va['loss'])
        history['train_char_acc'].append(tr['char_acc'])
        history['val_char_acc'].append(va['char_acc'])
        history['train_plate_acc'].append(tr['plate_acc'])
        history['val_plate_acc'].append(va['plate_acc'])

        print(f"{epoch+1:>6} | {tr['loss']:>9.4f} {va['loss']:>9.4f} | "
              f"{tr['char_acc']:>8.2%} {va['char_acc']:>8.2%} | "
              f"{tr['plate_acc']:>8.2%} {va['plate_acc']:>8.2%} | "
              f"{elapsed:>5.1f}s")

        save_training_plots(history, epoch, layout, plots_dir)

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_plate_acc': best_plate_acc, 'history': history,
                'layout': layout, 'num_classes_per_pos': n_classes,
            }, ckpt_dir / f'epoch_{epoch+1:04d}.pth')

        if va['plate_acc'] > best_plate_acc:
            best_plate_acc = va['plate_acc']
            no_improve     = 0
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_plate_acc': best_plate_acc, 'history': history,
                'layout': layout, 'num_classes_per_pos': n_classes,
            }, best_ckpt_path)
            print(f"         ★ Novo melhor modelo — plate_acc val: {best_plate_acc:.2%}")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"\n[EarlyStopping] {args.patience} épocas sem melhora. Parando.")
            break

    # ── Pós-treinamento ────────────────────────────────────────────────────────
    print("\n[Pós-treino] Carregando melhor modelo...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])

    save_sample_predictions(model, val_loader, device, layout, plots_dir)
    print(f"[Plots]   Gráficos salvos em '{plots_dir}/'")

    onnx_path = export_dir / f'plate_ocr_{layout.lower()}.onnx'
    export_onnx(model, onnx_path, device)

    print(f"\n[Concluído]")
    print(f"  Melhor plate_acc (val): {best_plate_acc:.2%}")
    print(f"  Modelo ONNX:            {onnx_path}")
    print(f"  Gráficos:               {plots_dir}/")
    print(f"  Checkpoints:            {ckpt_dir}/")


if __name__ == '__main__':
    main()