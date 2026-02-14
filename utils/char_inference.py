from utils.inferencia import ONNXCharModel
import numpy as np

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

models_dict = {}

for tipo in ['numeros','letras']:
    for placa in ['br','me']:
        model_path = f'onnx_models/{placa}-{tipo}.onnx'
        models_dict[f'{placa}_{tipo}'] = ONNXCharModel(model_path)

def infer_characters(images, model_type, placa):
    model = models_dict[f'{placa}_{model_type}']
    results = model(images)[0]
    results = np.exp(results) / np.sum(np.exp(results), axis=1, keepdims=True)
    if model_type == 'numeros':
        return [(str(int(np.argmax(p))),p.max()) for p in results]
    else:
        return [(alphabet[np.argmax(p)],p.max()) for p in results]