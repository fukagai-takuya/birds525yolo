import sys
from ultralytics import YOLO

def main(args):
    if len(args) != 5:
        print('Usage:', args[0], 'path/to/training_dataset/data.yaml', 'pretrained_model:(yolov8n.pt etc.)', 'epochs:int', 'imgsz:int')
        return

    # Load a model
    model = YOLO(args[2])  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data=args[1], epochs=int(args[3]), imgsz=int(args[4]))
        
    return


if __name__ == "__main__":
    args = sys.argv
    main(args)
