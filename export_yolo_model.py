from ultralytics import YOLO
import shutil
import os

last_train = sorted(os.listdir('runs/pose'))[-1]
model = YOLO(f'runs/pose/{last_train}/weights/best.pt')
model.export(
    format='onnx',
    opset=17,
    dynamic=False,
    simplify=True,
    imgsz=640,
    half=False,
    verbose=True
)
shutil.move(f'runs/pose/{last_train}/weights/best.onnx', 'onnx_models/yolo26-pose.onnx')
print(f"Model exported to onnx_models/yolo26-pose.onnx")