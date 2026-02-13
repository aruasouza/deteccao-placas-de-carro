import numpy as np
from transform import *
from char_inference import infer_characters

LETRAS_INDEX = [0,1,2,4]
NUMEROS_INDEX = [3,5,6]

def crop(img):
    return img[60:190,30:470]

def pipeline(img,percentile = 35):
    img = crop(img)
    img = trashold(img, percentile)
    splited_chars = [clean_objects(char) for char in split_chars(img,7)]
    letters = [clean_objects(splited_chars[i]) for i in LETRAS_INDEX]
    numbers = [clean_objects(splited_chars[i]) for i in NUMEROS_INDEX]
    infered_letters = infer_characters(letters, 'letras', 'me')
    infered_numbers = infer_characters(numbers, 'numeros', 'me')
    placa = ['*'] * 7
    for i,index in enumerate(LETRAS_INDEX):
        placa[index] = infered_letters[i]
    for i,index in enumerate(NUMEROS_INDEX):
        placa[index] = infered_numbers[i]
    return ''.join(placa)