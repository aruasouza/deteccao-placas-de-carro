import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from utils.train_letter_models import CharCNN
from utils.train_number_models import DigitCNN

number_model_br = DigitCNN()
model_path = f'charmodels/br-numeros.pth'
number_model_br.load_state_dict(torch.load(model_path, map_location='cpu'))
number_model_br.eval()

number_model_me = DigitCNN()
model_path = f'charmodels/me-numeros.pth'
number_model_me.load_state_dict(torch.load(model_path, map_location='cpu'))
number_model_me.eval()

letter_model_br = CharCNN()
model_path = f'charmodels/br-letras.pth'
letter_model_br.load_state_dict(torch.load(model_path, map_location='cpu'))
letter_model_br.eval()

letter_model_me = CharCNN()
model_path = f'charmodels/me-letras.pth'
letter_model_me.load_state_dict(torch.load(model_path, map_location='cpu'))
letter_model_me.eval()

def infer_characters(images, model_type='numeros', placa='br'):

    if model_type == 'numeros':
        is_digit = True
        if placa == 'br':
            model = number_model_br
        if placa == 'me':
            model = number_model_me
    elif model_type == 'letras':
        is_digit = False
        if placa == 'br':
            model = letter_model_br
        elif placa == 'me':
            model = letter_model_me
    else:
        raise ValueError("model_type must be 'numeros' or 'letras'")
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    tensors = [transform(np.array(img * 255, dtype=np.uint8)) for img in images]
    batch = torch.stack(tensors)
    
    with torch.no_grad():
        outputs = model(batch)
        probabilities = torch.softmax(outputs, dim=1).numpy()
        _, predicted = torch.max(outputs, 1)
    
    preds = predicted.cpu().numpy()
    
    if is_digit:
        return [(str(p),prob.max()) for p,prob in zip(preds,probabilities)]
    else:
        return [(chr(p + ord('A')),prob.max()) for p,prob in zip(preds,probabilities)]