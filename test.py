import os
import cv2
import numpy as np
from utils.main_pipeline import full_pipeline

def test_pipeline():
    test_dir = 'datasets/test'
    results = {
        'corretos': 0, 
        'total': 0, 
        'tempo_deteccao': [],
        'tempo_extracao': [],
        'tempo_leitura': [],
        'tempo_total': []
    }
    
    for filename in os.listdir(test_dir):
        if not filename.endswith('.jpg'):
            continue
        
        placa_esperada = filename.replace('.jpg', '')
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path)
        
        resultado = full_pipeline(img)
        results['total'] += 1
        
        if resultado:
            placa_lida, confianca = resultado['leitura']
            tempos = resultado['tempo']
            
            results['tempo_deteccao'].append(tempos['tempo_deteccao'])
            results['tempo_extracao'].append(tempos['tempo_extracao'])
            results['tempo_leitura'].append(tempos['tempo_leitura'])
            results['tempo_total'].append(sum(tempos.values()))
            
            if placa_lida == placa_esperada:
                results['corretos'] += 1
    
    acuracia = (results['corretos'] / results['total']) * 100
    
    print(f"\n{'='*50}")
    print(f"RELATÓRIO DE TESTE - DETECÇÃO DE PLACAS")
    print(f"{'='*50}")
    print(f"Total de imagens: {results['total']}")
    print(f"Acertos: {results['corretos']}")
    print(f"Erros: {results['total'] - results['corretos']}")
    print(f"Acurácia: {acuracia:.2f}%")
    print(f"\n{'='*50}")
    print(f"TEMPOS DE INFERÊNCIA (ms)")
    print(f"{'='*50}")
    print(f"Detecção    - Média: {np.mean(results['tempo_deteccao'])*1000:.2f}ms")
    print(f"Extração    - Média: {np.mean(results['tempo_extracao'])*1000:.2f}ms")
    print(f"Leitura     - Média: {np.mean(results['tempo_leitura'])*1000:.2f}ms")
    print(f"Total       - Média: {np.mean(results['tempo_total'])*1000:.2f}ms")
    print(f"{'='*50}\n")
    
    return results

if __name__ == '__main__':
    test_pipeline()
