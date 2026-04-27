from dotenv import load_dotenv
load_dotenv()

import os
import cv2
import numpy as np
import utils.main_pipeline as main
import utils.br_pipeline as br
import utils.me_pipeline as me

def test_pipeline():
    test_dir = 'datasets/test'
    
    for filename in os.listdir(test_dir):
        if not filename.endswith('.jpg'):
            continue
        
        placa_esperada = filename.replace('.jpg', '')
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path)
        
        print(f'Placa esperada: {placa_esperada}')
        outputs, (orig_h, orig_w) = main.model(img)
        print(f'Origem: {orig_h}, {orig_w}')
        detections = outputs[0][0]
        # print(list(detections[0]))
        deteccao = main.detect_placa(main.model, img)
        if deteccao is None:
            print('Nenhuma placa detectada')
            continue
        print(f'Detecção: {deteccao}')
        gray = main.grayscale(img)
        croped_img = main.get_croped_image(deteccao, gray)
        warped_img = main.warp_image(deteccao, croped_img)
        if deteccao['classe'] == 1:
            print('Iniciando pipeline de placas BR')
            extraction = br.extract_pipeline(warped_img, 30)
            infered_letters = br.infer_characters(extraction['letras'], model_type='letras', placa='br')
            infered_numbers = br.infer_characters(extraction['numeros'], model_type='numeros', placa='br')
            confianca = 1.0
            for inf in infered_letters + infered_numbers:
                confianca *= inf[1]
            print(''.join([inf[0] for inf in infered_letters] + [inf[0] for inf in infered_numbers]),confianca)
        else:
            print('Iniciando pipeline de placas ME')
            extraction = me.extract_pipeline(warped_img, 35)
            infered_letters = me.infer_characters(extraction['letras'], 'letras', 'me')
            infered_numbers = me.infer_characters(extraction['numeros'], 'numeros', 'me')
            placa = ['*'] * 7
            confianca = 1.0
            for i,index in enumerate(me.LETRAS_INDEX):
                placa[index] = infered_letters[i][0]
                confianca *= infered_letters[i][1]
            for i,index in enumerate(me.NUMEROS_INDEX):
                placa[index] = infered_numbers[i][0]
                confianca *= infered_numbers[i][1]
            print(''.join(placa),confianca)

if __name__ == '__main__':
    test_pipeline()
