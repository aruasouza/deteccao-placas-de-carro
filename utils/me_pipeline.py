import numpy as np
from utils.transform import *
from utils.char_inference import infer_characters

LETRAS_INDEX = [0,1,2,4]
NUMEROS_INDEX = [3,5,6]

def crop(img):
    return img[60:190,40:460]

def pipeline(img,percentile = 35):
    img = crop(img)
    img = trashold(img, percentile)
    splited_chars = [clean_objects(char) for char in split_chars(img,7)]
    letters = [clean_objects(splited_chars[i]) for i in LETRAS_INDEX]
    numbers = [clean_objects(splited_chars[i]) for i in NUMEROS_INDEX]
    infered_letters = infer_characters(letters, 'letras', 'me')
    infered_numbers = infer_characters(numbers, 'numeros', 'me')
    placa = ['*'] * 7
    confianca = 1.0
    for i,index in enumerate(LETRAS_INDEX):
        placa[index] = infered_letters[i][0]
        confianca *= infered_letters[i][1]
    for i,index in enumerate(NUMEROS_INDEX):
        placa[index] = infered_numbers[i][0]
        confianca *= infered_numbers[i][1]
    return ''.join(placa),confianca