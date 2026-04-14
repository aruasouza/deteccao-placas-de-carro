from rknn.api import RKNN
from pathlib import Path

ONNX_DIR = Path('onnx_models')
RKNN_DIR = Path('/home/ubuntu/rknn_models')
DATASET_DIR = Path('datasets/rknn_datasets')
QUANT = True
QUANTYOLO = True

RKNN_DIR.mkdir(exist_ok=True)

models = [
    ('br-letras.onnx', 'br-letras_dataset.txt'),
    ('br-numeros.onnx', 'br-numeros_dataset.txt'),
    ('me-letras.onnx', 'me-letras_dataset.txt'),
    ('me-numeros.onnx', 'me-numeros_dataset.txt'),
    ('yolo-pose.onnx', 'yolo_dataset.txt'),
]

for onnx_file, dataset_file in models:
    print(f'\n{"="*60}')
    print(f'Convertendo: {onnx_file}')
    print(f'{"="*60}')
    
    rknn = RKNN(verbose=True)
    
    if 'yolo' not in onnx_file:
        rknn.config(
            target_platform='rk3568',
            quantized_dtype='asymmetric_quantized-8',
            quantized_algorithm='normal',
            quantized_method='layer',
            mean_values=[[127.5]],
            std_values=[[127.5]],
            optimization_level=3
        )
        if rknn.load_onnx(model=str(ONNX_DIR / onnx_file),input_size_list=[[1, 1, 28, 28]]) != 0:
            print(f'Falha ao carregar {onnx_file}')
            continue

        if rknn.build(do_quantization=QUANT, dataset=str(DATASET_DIR / dataset_file)) != 0:
            print(f'Falha no build de {onnx_file}')
            continue

    else:
        rknn.config(
            target_platform='rk3568',
            quantized_dtype='asymmetric_quantized-8',
            quantized_algorithm='normal',
            quantized_method='layer',
            mean_values=[[0.0, 0.0, 0.0]],
            std_values=[[255.0, 255.0, 255.0]],
            optimization_level=3
        )
        if rknn.load_onnx(model=str(ONNX_DIR / onnx_file),input_size_list=[[1, 3, 640, 640]]) != 0:
            print(f'Falha ao carregar {onnx_file}')
            continue
    
        if rknn.build(do_quantization=QUANTYOLO, dataset=str(DATASET_DIR / dataset_file)) != 0:
            print(f'Falha no build de {onnx_file}')
            continue
    
    rknn_output = RKNN_DIR / onnx_file.replace('.onnx', '.rknn')
    if rknn.export_rknn(str(rknn_output)) != 0:
        print(f'Falha ao exportar {onnx_file}')
        continue
    
    rknn.release()
    print(f'✓ {onnx_file} convertido com sucesso!')

print('\n' + '='*60)
print('Conversão concluída! Copie os arquivos .rknn para a Rock 3A.')
print('='*60)