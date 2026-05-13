import os
import time
from utils.inferencia_ocr import PlateOCR

env = os.getenv('ENV', 'DEV')

# if env == 'PROD':
#     ocr_model_me = PlateOCR('rknn_models/plate_ocr_me.rknn')
#     ocr_model_br = PlateOCR('rknn_models/plate_ocr_br.rknn')
# else:
ocr_model_me = PlateOCR('onnx_models/plate_ocr_me.onnx')
ocr_model_br = PlateOCR('onnx_models/plate_ocr_br.onnx')

def inferencia_me(img):
    result = {'tempo':{'tempo_extracao':0}}
    start = time.time()
    leitura = ocr_model_me((img * 255).astype('uint8'), layout='ME')
    result['tempo']['tempo_leitura'] = time.time() - start
    result['leitura'] = leitura
    return result

def inferencia_br(img):
    result = {'tempo':{'tempo_extracao':0}}
    start = time.time()
    leitura = ocr_model_br((img * 255).astype('uint8'), layout='BR')
    result['tempo']['tempo_leitura'] = time.time() - start
    result['leitura'] = leitura
    return result