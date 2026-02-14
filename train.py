from ultralytics import YOLO
import shutil
import os

EPOCHS = 100

if __name__ == '__main__':
    model = YOLO('yolo26n-pose.pt')
    model.train(
        data='dataset.yaml', 
        epochs=EPOCHS,
        batch=-1,
        device='cuda',
    )
    last_train = sorted(os.listdir('runs/pose/train/weights'))[-1]
    model = YOLO(f'runs/pose/{last_train}/weights/best.pt')
    model.export(format='onnx')
    shutil.move(f'runs/pose/{last_train}/weights/best.onnx', 'onnx_models/best.onnx')