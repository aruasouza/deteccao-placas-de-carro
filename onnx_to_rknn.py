from rknn.api import RKNN

rknn = RKNN(verbose=True)
rknn.config(target_platform='rk3568')

ret = rknn.load_onnx(model='seu_modelo.onnx')
if ret != 0:
    print('Falha ao carregar ONNX')
    exit(ret)

ret = rknn.build(do_quantization=True,
                 dataset='dataset.txt')
if ret != 0:
    print('Falha no build')
    exit(ret)

ret = rknn.export_rknn('modelo.rknn')
if ret != 0:
    print('Falha ao exportar')
    exit(ret)

rknn.release()
print("Conversão concluída! Copie 'modelo.rknn' para a Rock 3A.")