import torch
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

models_dict = {
    'br': {
            'numeros': number_model_br,
            'letras': letter_model_br
        },
        'me': {
            'numeros': number_model_me,
            'letras': letter_model_me
        }
}

dummy_input = torch.randn(1, 1, 28, 28)

for tipo in ['numeros','letras']:
    for placa in ['br','me']:
        onnx_path = f'onnx_models/{placa}-{tipo}.onnx'

        torch.onnx.export(
            models_dict[placa][tipo],
            dummy_input,
            onnx_path,
            dynamo=True,
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input":  {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            export_params=True,
            do_constant_folding=True,
            verbose=False,
        )

        print(f"Model exported to {onnx_path}")