import numpy as np
import os

env = os.getenv('ENV', 'DEV')

if env == 'PROD':
    from utils.inferencia import RKNNCharModel as CharModel
    model_dir = 'rknn_models'
    model_ext = 'rknn'
else:
    from utils.inferencia import ONNXCharModel as CharModel
    model_dir = 'onnx_models'
    model_ext = 'onnx'

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

models_dict = {}

for tipo in ['numeros','letras']:
    for placa in ['br','me']:
        model_path = f'{model_dir}/{placa}-{tipo}.{model_ext}'
        models_dict[f'{placa}_{tipo}'] = CharModel(model_path)

def infer_characters(images, model_type, placa):
    model = models_dict[f'{placa}_{model_type}']
    results = model(images)[0]
    results = np.exp(results) / np.sum(np.exp(results), axis=1, keepdims=True)
    if model_type == 'numeros':
        return [(str(int(np.argmax(p))),p.max()) for p in results]
    else:
        return [(alphabet[np.argmax(p)],p.max()) for p in results]