from utils.train_letter_models import main as train_letter
from utils.train_number_models import main as train_number

EPOCHS = 20

for tipo in ['me','br']:
    print(f'Treinando modelo de números para placas {tipo.upper()}')
    train_number(PLACA = tipo,EPOCHS = EPOCHS)
    print(f'Treinando modelo de letras para placas {tipo.upper()}')
    train_letter(PLACA = tipo, EPOCHS = EPOCHS)