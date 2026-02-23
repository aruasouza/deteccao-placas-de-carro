from ultralytics import YOLO

EPOCHS = 100

if __name__ == '__main__':
    model = YOLO('yolo26n-pose.pt')
    model.train(
        data='dataset.yaml', 
        epochs=EPOCHS,
        batch=-1,
        device='cuda',
    )