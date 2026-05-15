"""
inferencia_ocr.py — Inferência do modelo OCR de placas veiculares.

Backend selecionado pela variável de ambiente ENV:
    DEV  (padrão) — ONNX Runtime, CPU
    CUDA          — ONNX Runtime, GPU CUDA
    PROD          — RKNNLite, NPU Rock 3A

Uso:
    from utils.inferencia_ocr import PlateOCR

    # DEV / CUDA
    ocr = PlateOCR('onnx_models/plate_ocr_br.onnx')

    # PROD
    ocr = PlateOCR('rknn_models/plate_ocr_br.rknn')

    placa, conf = ocr(warped_gray_img, layout='BR')
    print(placa, conf)   # ex: "ODE2510"  0.923

A confiança é o produto das probabilidades softmax do caractere escolhido
em cada posição, resultando em um valor 0-1.
"""

from __future__ import annotations
import os
import numpy as np
import cv2

# ─────────────────────────── Vocabulário ──────────────────────────────────────

LETRAS  = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DIGITOS = '0123456789'

LAYOUT_VOCAB: dict[str, list[str]] = {
    'BR': [LETRAS, LETRAS, LETRAS, DIGITOS, DIGITOS, DIGITOS, DIGITOS],
    'ME': [LETRAS, LETRAS, LETRAS, DIGITOS, LETRAS,  DIGITOS, DIGITOS],
}

IMG_W, IMG_H = 100, 32
NUM_CHARS    = 7

ENV = os.getenv('ENV', 'DEV')

# ─────────────────────────── Helpers ──────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    """Softmax numericamente estável ao longo do último eixo."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _decode(logits_7: np.ndarray, layout: str) -> tuple[str, float]:
    """
    Decodifica (7, max_classes) de logits em (placa, confiança).

    Os logits de posições com menos classes foram preenchidos com -inf
    na exportação, então o softmax os zerará naturalmente.
    """
    vocabs = LAYOUT_VOCAB.get(layout, LAYOUT_VOCAB['BR'])
    probs  = _softmax(logits_7)   # (7, max_classes)

    plate_chars = []
    confidence  = 1.0

    for pos in range(NUM_CHARS):
        vocab     = vocabs[pos]
        n_classes = len(vocab)
        pos_probs = probs[pos, :n_classes]   # só as classes válidas da posição

        idx  = int(pos_probs.argmax())
        char = vocab[idx]

        plate_chars.append(char)
        confidence *= float(pos_probs[idx])

    return ''.join(plate_chars), confidence


def _preprocess(img: np.ndarray) -> np.ndarray:
    """
    Aceita grayscale HxW ou HxWx1, uint8 ou float32 0-1.
    Retorna float32 (1, 1, IMG_H, IMG_W) normalizado.
    """
    if img.ndim == 3:
        img = img[:, :, 0]
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img[np.newaxis, np.newaxis, :, :]   # (1, 1, H, W)


# ─────────────────────────── Backends ─────────────────────────────────────────

class _ONNXBackend:
    """ONNX Runtime — CPU (DEV) ou CUDA."""

    def __init__(self, path: str):
        import onnxruntime as ort
        if ENV == 'CUDA':
            import torch
            self.session    = ort.InferenceSession(path, providers=['CUDAExecutionProvider'])
        else:
            self.session    = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def run(self, tensor: np.ndarray) -> np.ndarray:
        """Retorna logits (B, 7, max_classes)."""
        [logits] = self.session.run(None, {self.input_name: tensor})
        return logits

    def run_batch(self, tensors: np.ndarray) -> np.ndarray:
        return self.run(tensors)


class _RKNNBackend:
    """RKNNLite — NPU Rock 3A (PROD)."""

    def __init__(self, path: str):
        from rknnlite.api import RKNNLite
        self.rknn = RKNNLite()
        self.rknn.load_rknn(path)
        self.rknn.init_runtime()

    def run(self, tensor: np.ndarray) -> np.ndarray:
        """
        RKNNLite espera uint8 sem dimensão de batch.
        Converte float32 (1,1,H,W) → uint8 (1,H,W) e roda inferência.
        Retorna logits (1, 7, max_classes).
        """
        inp = (tensor[0] * 255).astype(np.uint8)   # (1, H, W)
        [logits] = self.rknn.inference(inputs=[inp])
        return logits   # já (1, 7, max_classes) conforme exportação

    def run_batch(self, tensors: np.ndarray) -> np.ndarray:
        """RKNNLite não suporta batch nativo — itera uma a uma."""
        results = [self.run(tensors[b:b+1]) for b in range(tensors.shape[0])]
        return np.concatenate(results, axis=0)


# ─────────────────────────── PlateOCR ─────────────────────────────────────────

class PlateOCR:
    """
    Wrapper de OCR de placas com backend selecionado por ENV.

    Parâmetros
    ----------
    path : str
        .onnx para ENV=DEV ou ENV=CUDA
        .rknn para ENV=PROD
    """

    def __init__(self, path: str):
        if ENV == 'PROD':
            # self._backend = _RKNNBackend(path)
            self._backend = _ONNXBackend(path)
        else:
            self._backend = _ONNXBackend(path)

    def __call__(self, img: np.ndarray,
                 layout: str = 'BR') -> tuple[str, float]:
        """
        Reconhece a placa em `img`.

        Parâmetros
        ----------
        img    : array HxW (grayscale), já recortado e retificado.
        layout : 'BR' ou 'ME'.

        Retorna
        -------
        plate : string de 7 caracteres  (ex: "ODE2510")
        conf  : confiança 0-1           (ex: 0.923)
        """
        tensor = _preprocess(img)
        logits = self._backend.run(tensor)   # (1, 7, max_classes)
        return _decode(logits[0], layout)

    def batch(self, imgs: list[np.ndarray],
              layout: str = 'BR') -> list[tuple[str, float]]:
        """
        Processa um lote de imagens.

        Retorna lista de (plate, conf) na mesma ordem das imagens.
        """
        tensors = np.concatenate([_preprocess(im) for im in imgs], axis=0)
        logits  = self._backend.run_batch(tensors)   # (B, 7, max_classes)
        return [_decode(logits[b], layout) for b in range(logits.shape[0])]


# ─── Integração com o pipeline existente ──────────────────────────────────────

def integrate_with_pipeline(warped_img: np.ndarray,
                             ocr: PlateOCR,
                             classe: int) -> tuple[str, float]:
    """
    Adaptor para uso direto no full_pipeline de main_pipeline.py.

    Parâmetros
    ----------
    warped_img : imagem retificada em escala de cinza (float32 0-1 ou uint8)
    ocr        : instância de PlateOCR
    classe     : 0 = ME, 1 = BR  (convenção do modelo YOLO-pose)

    Retorna
    -------
    (plate, conf)
    """
    layout = 'BR' if classe == 1 else 'ME'
    return ocr(warped_img, layout=layout)


# ─── Teste rápido ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    path     = sys.argv[1] if len(sys.argv) > 1 else 'onnx_models/plate_ocr_br.onnx'
    img_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"[ENV={ENV}] Carregando {path}...")
    ocr = PlateOCR(path)

    if img_path:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Erro: não foi possível carregar '{img_path}'")
            sys.exit(1)
        plate, conf = ocr(img)
        print(f"Placa: {plate}  |  Confiança: {conf:.4f} ({conf:.1%})")
    else:
        dummy = np.random.rand(IMG_H, IMG_W).astype(np.float32)
        plate, conf = ocr(dummy)
        print(f"Teste com imagem aleatória: {plate}  |  Confiança: {conf:.4f}")
        print("Modelo carregado com sucesso.")