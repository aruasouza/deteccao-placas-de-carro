"""
prepare_dataset.py — Pré-processamento offline do dataset de placas.

Para cada imagem bruta de carro:
  1. Roda o modelo YOLO-pose para detectar a placa e seus keypoints
  2. Recorta e retifica a perspectiva (warp_image)
  3. Converte para escala de cinza
  4. Salva a imagem resultante em datasets/dataset1/placas/<split>/

As imagens processadas ficam com o mesmo nome da original, permitindo que o
train_plate_ocr.py continue usando o índice de anotações existente.

Uso:
    python prepare_dataset.py [--yolo onnx_models/yolo-pose.onnx] [--minconf 0.3]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Importa o pipeline existente do projeto
sys.path.insert(0, str(Path(__file__).parent))
from utils.inferencia import ONNXYOLO

# ─────────────────────────── Caminhos ─────────────────────────────────────────

DATASETS_ROOT = Path('datasets/dataset1')

# Imagens brutas (fotos de carros)
SPLITS = {
    'train': DATASETS_ROOT / 'preparado' / 'images' / 'train',
    'val':   DATASETS_ROOT / 'preparado' / 'images' / 'val',
}

# Destino das placas recortadas
OUTPUT_ROOT = DATASETS_ROOT / 'placas'

# Dimensão da saída do warp (deve coincidir com IMG_W, IMG_H do script de treino)
WARP_W, WARP_H = 500, 200

# ─────────────────────────── Funções do pipeline ──────────────────────────────
# Reproduzidas aqui para não depender de main_pipeline.py diretamente.

def detect_placa(model, img, minconf=0.0):
    outputs, (orig_h, orig_w) = model(img)
    if not outputs or len(outputs) == 0:
        return None
    input_h, input_w = model.input_shape[2:]
    scale_x = orig_w / input_w
    scale_y = orig_h / input_h
    detections = outputs[0][0]
    if len(detections) == 0:
        return None
    best_det = detections[0].T
    x1, y1, x2, y2 = best_det[:4]
    conf = best_det[4]
    if conf < minconf:
        return None
    classe  = int(best_det[5])
    kp_data = best_det[6:]
    kp  = [(int(kp_data[i] * scale_x), int(kp_data[i + 1] * scale_y))
           for i in range(0, len(kp_data), 3)]
    box = [int(x1 * scale_x), int(y1 * scale_y),
           int(x2 * scale_x), int(y2 * scale_y)]
    return {'classe': classe, 'box': box, 'kp': kp}


def transform_keypoints(resposta):
    x1, y1, x2, y2 = resposta['box']
    return [
        (min(nk[0] - x1, x2 - x1 - 1), min(nk[1] - y1, y2 - y1 - 1))
        for nk in resposta['kp']
    ]


def warp_image(resposta, img):
    src_points = transform_keypoints(resposta)
    dst_points = [(0, 0), (WARP_W - 1, 0), (WARP_W - 1, WARP_H - 1), (0, WARP_H - 1)]
    src    = np.float32(src_points)
    dst    = np.float32(dst_points)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (WARP_W, WARP_H))


def process_image(img_path: Path, model, minconf: float):
    """
    Lê a imagem, detecta a placa, retifica e retorna array grayscale (H, W) uint8.
    Retorna None se a detecção falhar.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None, 'leitura falhou'

    deteccao = detect_placa(model, img, minconf)
    if deteccao is None:
        return None, 'sem detecção'

    # Recortar bounding box e retificar perspectiva em escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = deteccao['box']
    cropped = gray[y1:y2, x1:x2]

    try:
        warped = warp_image(deteccao, cropped)
    except Exception as e:
        return None, f'warp falhou: {e}'

    if warped is None or warped.size == 0:
        return None, 'warp vazio'

    return warped, None


# ─────────────────────────── Main ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Pré-processar dataset de placas')
    p.add_argument('--yolo',    type=str, default='onnx_models/yolo-pose.onnx',
                   help='Caminho do modelo YOLO-pose ONNX')
    p.add_argument('--minconf', type=float, default=0.01,
                   help='Confiança mínima de detecção (0-1)')
    p.add_argument('--overwrite', action='store_true',
                   help='Reprocessar imagens já existentes no destino')
    return p.parse_args()


def main():
    args = parse_args()

    yolo_path = Path(args.yolo)
    if not yolo_path.exists():
        print(f"[Erro] Modelo YOLO não encontrado: {yolo_path}")
        sys.exit(1)

    print(f"[YOLO] Carregando {yolo_path}...")
    model = ONNXYOLO(str(yolo_path))
    print(f"       Input shape: {model.input_shape}  |  minconf: {args.minconf}")

    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    total_ok = total_fail = 0

    for split, src_dir in SPLITS.items():
        if not src_dir.exists():
            print(f"[Aviso] Diretório não encontrado, pulando: {src_dir}")
            continue

        dst_dir = OUTPUT_ROOT / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        img_paths = [p for p in src_dir.iterdir()
                     if p.suffix.lower() in img_extensions]

        print(f"\n[{split.upper()}] {len(img_paths)} imagens → {dst_dir}")

        ok = fail = skip = 0
        for img_path in img_paths:
            dst_path = dst_dir / img_path.name

            if dst_path.exists() and not args.overwrite:
                skip += 1
                continue

            warped, err = process_image(img_path, model, 0)

            if warped is None:
                fail += 1
                print(f"  ✗ {img_path.name}: {err}")
                continue

            cv2.imwrite(str(dst_path), warped)
            ok += 1

        total_ok   += ok
        total_fail += fail
        print(f"  Salvas: {ok}  |  Falhas: {fail}  |  Puladas: {skip}")

    print(f"\n[Concluído] OK: {total_ok}  |  Falhas: {total_fail}")
    print(f"Imagens salvas em: {OUTPUT_ROOT}/")
    print("\nAgora treine com:")
    print("  python train_plate_ocr.py --layout BR")
    print("  python train_plate_ocr.py --layout ME")


if __name__ == '__main__':
    main()