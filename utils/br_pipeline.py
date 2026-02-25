import numpy as np
import time
from utils.transform import *
from utils.char_inference import infer_characters

def crop(img):
    return img[40:190, 10:490]

def letters_numbers_split(placa):
    letters = placa[:,:200]
    numbers = placa[:,210:]
    return letters,numbers

def detect_reversed(img):
    flat = img.ravel()
    median = np.percentile(flat, 50)
    per95 = np.percentile(flat, 95)
    per5 = np.percentile(flat, 5)
    return (median - per5) < (per95 - median)

def extract_pipeline(img, percentile = 30):
    img = crop(img)
    rev = detect_reversed(img)
    placa = trashold(img, percentile, reverse = rev)
    letters,numbers = letters_numbers_split(placa)
    splitted_letters = [clean_objects(let) for let in split_chars(letters, 3)]
    splitted_numbers = [clean_objects(num) for num in split_chars(numbers, 4)]
    return {'letras':splitted_letters,'numeros':splitted_numbers}

def read_chars_pipeline(extraction):
    infered_letters = infer_characters(extraction['letras'], model_type='letras', placa='br')
    infered_numbers = infer_characters(extraction['numeros'], model_type='numeros', placa='br')
    confianca = 1.0
    for inf in infered_letters + infered_numbers:
        confianca *= inf[1]
    return ''.join([inf[0] for inf in infered_letters] + [inf[0] for inf in infered_numbers]),confianca

def pipeline(img,percentile = 30):
    result = {'tempo':{}}
    start = time.time()
    extraction = extract_pipeline(img, percentile)
    result['tempo']['tempo_extracao'] = time.time() - start
    start = time.time()
    leitura = read_chars_pipeline(extraction)
    result['tempo']['tempo_leitura'] = time.time() - start
    result['leitura'] = leitura
    return result