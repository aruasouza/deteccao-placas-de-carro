from utils.main_pipeline import *
import cv2
import os
import glob

def parse_anotation(full_path):
    with open(full_path) as f:
        content = f.readlines()
    return content[1].strip().split(': ')[1]

LETRAS_INDEX = [0,1,2,4]
NUMEROS_INDEX = [3,5,6]

ORIGEM = os.path.join('datasets','dataset1','bruto')
NAMES = os.path.join('datasets','dataset1','preparado','images')
DESTINO = os.path.join('datasets','dataset_chars')

images_br = sorted(glob.glob(os.path.join(ORIGEM, 'cars-br', '*.jpg')))
labels_br = sorted(glob.glob(os.path.join(ORIGEM, 'cars-br', '*.txt')))

images_me = sorted(glob.glob(os.path.join(ORIGEM, 'cars-me', '*.jpg')))
labels_me = sorted(glob.glob(os.path.join(ORIGEM, 'cars-me', '*.txt')))

index = 0
for direc in ['train','val','test']:
    print(f'Iniciando dataset "{direc}"')
    img_names = set(os.listdir(os.path.join(NAMES, direc)))
    dataset_me = [(img,label) for img,label in zip(images_me, labels_me) if os.path.basename(img) in img_names]
    dataset_br = [(img, label) for img, label in zip(images_br, labels_br) if os.path.basename(img) in img_names]
    destino = os.path.join(DESTINO, direc, 'br')
    destino_letras = os.path.join(destino, 'letras')
    os.makedirs(destino_letras, exist_ok=True)
    destino_numeros = os.path.join(destino, 'numeros')
    os.makedirs(destino_numeros, exist_ok=True)
    for img,label in dataset_br:
        plate = parse_anotation(label)
        chars = list(plate)
        image_chars = extract_characters(cv2.imread(img))
        if image_chars is None:
            continue
        letras = image_chars['letras']
        if len(letras) != 3:
            continue
        for i in range(3):
            cv2.imwrite(os.path.join(destino_letras, f'{chars[i]}_{index}.jpg'), (letras[i] * 255).astype('uint8'))
            index += 1
        numeros = image_chars['numeros']
        if len(numeros) != 4:
            continue
        for i in range(4):
            cv2.imwrite(os.path.join(destino_numeros, f'{chars[i+3]}_{index}.jpg'), (numeros[i] * 255).astype('uint8'))
            index += 1
        print(f'Processado: {index}')
    destino = os.path.join(DESTINO, direc, 'me')
    destino_letras = os.path.join(destino, 'letras')
    os.makedirs(destino_letras, exist_ok=True)
    destino_numeros = os.path.join(destino, 'numeros')
    os.makedirs(destino_numeros, exist_ok=True)
    for img,label in dataset_me:
        plate = parse_anotation(label)
        chars = list(plate)
        image_chars = extract_characters(cv2.imread(img))
        if image_chars is None:
            continue
        letras = image_chars['letras']
        if len(letras) != len(LETRAS_INDEX):
            continue
        for i,j in enumerate(LETRAS_INDEX):
            cv2.imwrite(os.path.join(destino_letras, f'{chars[j]}_{index}.jpg'), (letras[i] * 255).astype('uint8'))
            index += 1
        numeros = image_chars['numeros']
        if len(numeros) != len(NUMEROS_INDEX):
            continue
        for i,j in enumerate(NUMEROS_INDEX):
            cv2.imwrite(os.path.join(destino_numeros, f'{chars[j]}_{index}.jpg'), (numeros[i] * 255).astype('uint8'))
            index += 1
        print(f'Processado: {index}')
