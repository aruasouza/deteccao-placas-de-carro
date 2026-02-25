import os
import glob
import shutil

def parse_anotation(full_path):
    with open(full_path) as f:
        content = f.readlines()
    return content[1].strip().split(': ')[1]

ORIGEM = os.path.join('datasets','dataset1','bruto')
NAMES = os.path.join('datasets','dataset1','preparado','images')
destino = os.path.join('datasets','test')

images_br = sorted(glob.glob(os.path.join(ORIGEM, 'cars-br', '*.jpg')))
labels_br = sorted(glob.glob(os.path.join(ORIGEM, 'cars-br', '*.txt')))

images_me = sorted(glob.glob(os.path.join(ORIGEM, 'cars-me', '*.jpg')))
labels_me = sorted(glob.glob(os.path.join(ORIGEM, 'cars-me', '*.txt')))

index = 0
direc = 'test'
img_names = set(os.listdir(os.path.join(NAMES, direc)))
dataset_me = [(img,label) for img,label in zip(images_me, labels_me) if os.path.basename(img) in img_names]
dataset_br = [(img, label) for img, label in zip(images_br, labels_br) if os.path.basename(img) in img_names]
os.makedirs(destino, exist_ok=True)
for img,label in dataset_br:
    plate = parse_anotation(label)
    shutil.copy(img, os.path.join(destino, f'{plate}.jpg'))
    index += 1
    print(f'Processado: {index}')
for img,label in dataset_me:
    plate = parse_anotation(label)
    shutil.copy(img, os.path.join(destino, f'{plate}.jpg'))
    index += 1
    print(f'Processado: {index}')
