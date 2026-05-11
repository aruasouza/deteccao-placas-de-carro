import numpy as np
from pathlib import Path
from PIL import Image
import shutil
import os

SIZE = 500
YM = 2
OUTPUT_DIR = Path('datasets/rknn_datasets')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

yolo_images = list(Path('datasets/dataset1/preparado/images/train').glob('*.jpg'))[:SIZE * YM]
shutil.rmtree(OUTPUT_DIR / 'yolo',ignore_errors=True)
os.makedirs(OUTPUT_DIR / 'yolo', exist_ok=True)
with open(OUTPUT_DIR / 'yolo_dataset.txt', 'w') as f:
    for i, img_path in enumerate(yolo_images):
        jpg_path = OUTPUT_DIR / 'yolo' / f'sample_{i:04d}.jpg'
        shutil.copy(img_path, jpg_path)
        simple_path = 'yolo/' + f'sample_{i:04d}.jpg'
        f.write(f'{simple_path}\n')
print(f'Dataset criado: YOLO ({len(yolo_images)} amostras)')

placas_images = list(Path('datasets/dataset1/placas/train').glob('*.jpg'))[:SIZE * YM]
shutil.rmtree(OUTPUT_DIR / 'placas',ignore_errors=True)
os.makedirs(OUTPUT_DIR / 'placas', exist_ok=True)
with open(OUTPUT_DIR / 'placas_dataset.txt', 'w') as f:
    for i, img_path in enumerate(placas_images):
        jpg_path = OUTPUT_DIR / 'placas' / f'sample_{i:04d}.jpg'
        shutil.copy(img_path, jpg_path)
        simple_path = 'placas/' + f'sample_{i:04d}.jpg'
        f.write(f'{simple_path}\n')
print(f'Dataset criado: placas ({len(placas_images)} amostras)')

configs = [
    ('br-letras', 'datasets/dataset_chars/train/br/letras'),
    ('br-numeros', 'datasets/dataset_chars/train/br/numeros'),
    ('me-letras', 'datasets/dataset_chars/train/me/letras'),
    ('me-numeros', 'datasets/dataset_chars/train/me/numeros'),
]

for model_name, img_dir in configs:
    images = list(Path(img_dir).glob('*.jpg'))[:SIZE]
    shutil.rmtree(OUTPUT_DIR / model_name,ignore_errors=True)
    os.makedirs(OUTPUT_DIR / model_name, exist_ok=True)
    with open(OUTPUT_DIR / f'{model_name}_dataset.txt', 'w') as f:
        for i, img_path in enumerate(images):     
            jpg_path = OUTPUT_DIR / model_name / f'sample_{i:04d}.jpg'
            shutil.copy(img_path, jpg_path)
            simple_path = f'{model_name}/' + f'sample_{i:04d}.jpg'
            f.write(f'{simple_path}\n')
    
    print(f'Dataset criado: {model_name} ({len(images)} amostras)')

print('\nDatasets de quantização gerados')