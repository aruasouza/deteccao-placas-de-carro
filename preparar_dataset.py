import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil
import glob

PATH = os.path.join('datasets','dataset1','bruto')
DESTINO = os.path.join('datasets','dataset1','preparado')

classes = {
    'cars-me':0,
    'cars-br':1,
}

def parse_anotation(name,folder):
    full_path = os.path.join(PATH,folder, name + '.txt')
    with open(full_path) as f:
        content = f.readlines()
    tipe = content[0].strip().split(': ')[1]
    plate = content[1].strip().split(': ')[1]
    layout = content[2].strip().split(': ')[1]
    corners = content[3].strip().split(': ')[1]
    return tipe, plate, layout, corners

def load_image(name,folder):
    return np.array(Image.open(os.path.join(PATH, folder, name + '.jpg')))

def get_corners(anotation):
    tipe, plate, layout, corners = anotation
    corners = [x.split(',') for x in corners.split(' ')]
    return [(int(x[0]), int(x[1])) for x in corners]

def coco_anotation(classe,corners,img):
    img_height,img_width,_ = img.shape
    x_values = [x[0] / img_width for x in corners]
    y_values = [y[1] / img_height for y in corners]
    left = min(x_values)
    right = max(x_values)
    top = min(y_values)
    bottom = max(y_values)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    width = right - left
    height = bottom - top
    ann = f'{classe} {center_x} {center_y} {width} {height}'
    for x,y in zip(x_values, y_values):
        ann += f' {x} {y} 2'
    return ann

if __name__ == '__main__':

    folders = list(classes.keys())
    dataset = []
    for folder in folders:
        for image in os.listdir(os.path.join(PATH, folder)):
            if image.endswith('.jpg'):
                img_path = os.path.join(PATH, folder,image)
                name = image.split('.')[0]
                anot = parse_anotation(name, folder)
                corners = get_corners(anot)
                coco_ann = coco_anotation(classes[folder], corners, image)
                dataset.append((name, img_path, coco_ann))

    random.seed(0)
    random.shuffle(dataset)
    train = dataset[:7 * len(dataset) // 10]
    val = dataset[7 * len(dataset) // 10:9 * len(dataset) // 10]
    test = dataset[9 * len(dataset) // 10:]

    shutil.rmtree(DESTINO)

    part = 'train'
    labelsdir = os.path.join(DESTINO, 'labels', part)
    imagesdir = os.path.join(DESTINO, 'images', part)
    os.makedirs(labelsdir,exist_ok=True)
    os.makedirs(imagesdir,exist_ok=True)
    for item in train:
        shutil.copyfile(item[1], os.path.join(imagesdir, item[0] + '.jpg'))
        with open(os.path.join(labelsdir, item[0] + '.txt'), 'w') as f:
            f.write(item[2])

    part = 'val'
    labelsdir = os.path.join(DESTINO, 'labels', part)
    imagesdir = os.path.join(DESTINO, 'images', part)
    os.makedirs(labelsdir,exist_ok=True)
    os.makedirs(imagesdir,exist_ok=True)
    for item in train:
        shutil.copyfile(item[1], os.path.join(imagesdir, item[0] + '.jpg'))
        with open(os.path.join(labelsdir, item[0] + '.txt'), 'w') as f:
            f.write(item[2])

    part = 'test'
    labelsdir = os.path.join(DESTINO, 'labels', part)
    imagesdir = os.path.join(DESTINO, 'images', part)
    os.makedirs(labelsdir,exist_ok=True)
    os.makedirs(imagesdir,exist_ok=True)
    for item in train:
        shutil.copyfile(item[1], os.path.join(imagesdir, item[0] + '.jpg'))
        with open(os.path.join(labelsdir, item[0] + '.txt'), 'w') as f:
            f.write(item[2])


