from utils.train_letter_models import main as train_letter
from utils.train_number_models import main as train_number

EPOCHS = 20

for tipo in ['br','me']:
    train_number(PLACA = tipo,EPOCHS = EPOCHS)
    train_letter(PLACA = tipo, EPOCHS = EPOCHS)