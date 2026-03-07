import numpy as np
from pathlib import Path
from PIL import Image
import shutil

SIZE = 500
OUTPUT_DIR = Path('datasets/rknn_datasets')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

yolo_images = list(Path('datasets/dataset1/preparado/images/train').glob('*.jpg'))[:SIZE]
shutil.rmtree(OUTPUT_DIR / 'yolo')
with open(OUTPUT_DIR / 'yolo_dataset.txt', 'w') as f:
    for i, img_path in enumerate(yolo_images):
        pil_img = Image.open(img_path)
        pil_img = pil_img.resize((640, 640), Image.BILINEAR)
        img = np.array(pil_img)
        img = img.transpose(2, 0, 1)
        npy_path = OUTPUT_DIR / 'yolo' / f'sample_{i:04d}.npy'
        npy_path.parent.mkdir(exist_ok=True)
        np.save(npy_path, img[np.newaxis, ...])
        simple_path = 'yolo/' + f'sample_{i:04d}.npy'
        f.write(f'{simple_path}\n')
print(f'Dataset criado: YOLO ({len(yolo_images)} amostras)')

configs = [
    ('br-letras', 'datasets/dataset_chars/train/br/letras'),
    ('br-numeros', 'datasets/dataset_chars/train/br/numeros'),
    ('me-letras', 'datasets/dataset_chars/train/me/letras'),
    ('me-numeros', 'datasets/dataset_chars/train/me/numeros')
]

for model_name, img_dir in configs:
    images = list(Path(img_dir).glob('*.jpg'))[:SIZE]
    shutil.rmtree(OUTPUT_DIR / model_name)
    with open(OUTPUT_DIR / f'{model_name}_dataset.txt', 'w') as f:
        for i, img_path in enumerate(images):
            pil_img = Image.open(img_path).convert('L')
            pil_img = pil_img.resize((28, 28), Image.BILINEAR)
            img = np.array(pil_img)         
            npy_path = OUTPUT_DIR / model_name / f'sample_{i:04d}.npy'
            npy_path.parent.mkdir(exist_ok=True)
            np.save(npy_path, img[np.newaxis,np.newaxis, ...])
            simple_path = f'{model_name}/' + f'sample_{i:04d}.npy'
            f.write(f'{simple_path}\n')
    
    print(f'Dataset criado: {model_name} ({len(images)} amostras)')

print('\nDatasets de quantização gerados')