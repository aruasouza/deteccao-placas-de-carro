import numpy as np
import cv2
from pathlib import Path
from PIL import Image

SIZE = 500
OUTPUT_DIR = Path('datasets/rknn_datasets')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

yolo_images = list(Path('datasets/dataset1/preparado/images/train').glob('*.jpg'))[:SIZE]
with open(OUTPUT_DIR / 'yolo_dataset.txt', 'w') as f:
    for i, img_path in enumerate(yolo_images):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        npy_path = OUTPUT_DIR / 'yolo' / f'sample_{i:04d}.npy'
        npy_path.parent.mkdir(exist_ok=True)
        np.save(npy_path, img[np.newaxis, ...])
        f.write(f'{npy_path}\n')
print(f'Dataset criado: YOLO ({len(yolo_images)} amostras)')

configs = [
    ('br-letras', 'datasets/dataset_chars/train/br/letras'),
    ('br-numeros', 'datasets/dataset_chars/train/br/numeros'),
    ('me-letras', 'datasets/dataset_chars/train/me/letras'),
    ('me-numeros', 'datasets/dataset_chars/train/me/numeros')
]

for model_name, img_dir in configs:
    images = list(Path(img_dir).glob('*.jpg'))[:SIZE]
    with open(OUTPUT_DIR / f'{model_name}_dataset.txt', 'w') as f:
        for i, img_path in enumerate(images):
            img = Image.open(img_path).convert('L')
            img = img.resize((28, 28))
            img = np.array(img).astype(np.float32) / 255.0
            img = (img - 0.5) / 0.5
            
            npy_path = OUTPUT_DIR / model_name / f'sample_{i:04d}.npy'
            npy_path.parent.mkdir(exist_ok=True)
            np.save(npy_path, img[np.newaxis, ...])
            f.write(f'{npy_path}\n')
    
    print(f'Dataset criado: {model_name} ({len(images)} amostras)')

print('\nDatasets de quantização gerados')
