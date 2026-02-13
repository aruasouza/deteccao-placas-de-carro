from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo26n-pose.pt')
    model.train(
        data='dataset.yaml', 
        epochs=100,
        batch=-1, 
        device='cuda',
    )