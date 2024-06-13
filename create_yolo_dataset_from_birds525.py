import sys
import re
from pathlib import Path
from ultralytics import YOLO

NUMBER_OF_BIRD_SPECIES = 525

def update_train_valid_subdirecotry_dict(subdirectories, labels, subdir_dict):
    """
    Returns:
        (dictionary): keys are label names, values are directory objects

    Notes:
        The names of train, valid subdirectories will be used as category label names:

        The names of train and valid subdirectories has a slight difference like the following.
           - train/'PARAKETT  AUKLET'
           - valid/'PARAKETT AUKLET'

        So, update these names by eliminating duplicate space characters.
        Labels are sorted after eliminating duplicate space characters.
    """

    for subdir in subdirectories.iterdir():
        label_name = subdir.name.strip()
        label_name = re.sub('\s+', ' ', label_name)
        labels.append(label_name)
        subdir_dict[label_name] = subdir

    labels.sort()


def check_birds525_sub_dir(train_dir, valid_dir, label_names, train_dict, valid_dict):
    """
    Notes:
        The directory structure of train, valid subdirectories of kaggle birds 525 dataset:

            - archive/
                ├─ train/
                │   ├─ ABBOTTS BABBLER/
                │   ├─ ABBOTTS BOOBY/
                │   ├─ ...
                │   └─ ZEBRA DOVE/
                └─ valid/
                    ├─ ABBOTTS BABBLER/
                    ├─ ABBOTTS BOOBY/
                    ├─ ...
                    └─ ZEBRA DOVE/
    """

    # Update subdirectory names by eliminating duplicate space characters.
    # Label names are sorted after eliminating duplicate space characters.
    train_labels = []
    update_train_valid_subdirecotry_dict(train_dir, train_labels, train_dict)

    valid_labels = []
    update_train_valid_subdirecotry_dict(valid_dir, valid_labels, valid_dict)

    train_labels_length = len(train_labels)

    if train_labels_length != NUMBER_OF_BIRD_SPECIES:
        print('The train directory must have' , NUMBER_OF_BIRD_SPECIES, 'subdirectories')
        return False    
    
    if len(valid_labels) != train_labels_length:
        print('The train and the valid directories must have the same number of subdirectories')
        return False

    for i in range(train_labels_length):
        if (train_labels[i] != valid_labels[i]):
            print('The updated subdirectory names of the train and valid directories must be the same')
            print('train_labels[i]:', train_labels[i], ' valid_labels[i]:', valid_labels[i])
            return False

    # insert label names into empty label_names list
    label_names.extend(train_labels)
        
    return True


def check_birds525(args, label_names, train_dict, valid_dict):
    
    path_birds525 = Path(args[1])

    train_dir = None
    valid_dir = None

    for sub_dir in path_birds525.iterdir():
        if sub_dir.is_dir():
            if sub_dir.name == 'train':
                train_dir = sub_dir
            elif sub_dir.name == 'valid':
                valid_dir = sub_dir

    if train_dir is None or valid_dir is None:
        return False    
                
    if not check_birds525_sub_dir(train_dir, valid_dir, label_names, train_dict, valid_dict):
        return False
                
    return True


def output_ultralytics_yolo_format_data_yaml(path_output, label_names):
    file_name = "data.yaml"
    file_path = path_output / file_name
    with file_path.open("w", encoding ="utf-8") as fw:
        fw.write(f"path: {path_output.as_posix()}  # dataset root dir\n") # dataset root dir
        fw.write("train: images/train  # train images (relative to 'path')\n") # train images (relative to 'path')
        fw.write("train: images/val  # val images (relative to 'path')\n") # val images (relative to 'path')
        fw.write("\n") 
        fw.write("# Classes\n") 
        fw.write("names:\n")

        for i in range(len(label_names)):
            fw.write(f"{i}: {label_names[i]}\n")

    return


def output_ultralytics_yolo_formats_with_object_detection(label_names, input_dict, image_sub_dir, label_sub_dir, model):

    success_counter = 0
    failure_counter = 0

    for i in range(len(label_names)):
        label_name = label_names[i]
        input_dir = input_dict[label_name]

        for image_file in input_dir.iterdir():

            # Run inference with the YOLOv9c model on the image
            results = model(image_file)

            # Use a single image for each detection
            if len(results) != 1:
                print(f"Failed: len(results):{len(results)}, label_name:{label_name}, image_file:{image_file.name}")
                failure_counter += 1
                continue

            # Process results
            boxes = results[0].boxes  # Boxes object for bounding box outputs

            # Birds525 dataset has only one bird in each image.
            if len(boxes.cls) != 1:
                print(f"Failed: len(boxes.cls):{len(boxes.cls)}, label_name:{label_name}, image_file:{image_file.name}")
                failure_counter += 1
                continue

            # The bird class value is 14 when the pretrained network YOLO("yolov9c.pt") is used.            
            if boxes.cls[0] != 14:
                print(f"Failed: boxes.cls[0]:{boxes.cls[0]}, label_name:{label_name}, image_file:{image_file.name}")
                failure_counter += 1
                continue            

            success_counter += 1            
            # xywhn = boxes.xywhn[0]   # Normalized CenterX, CenterY, Width, Height
            # print(xywhn)
            # results[0].save(filename="result.jpg")  # save to disk

    print(f"success_counter: {success_counter}")
    print(f"failure_counter: {failure_counter}")
        
    return


def output_ultralytics_yolo_formats_val_train(label_names, input_dict, image_dir, label_dir, model, data_type):

    print(data_type)
    
    if data_type == "valid":
        image_sub_dir = image_dir / "val"
        label_sub_dir = label_dir / "val"
        
    elif data_type == "train":
        image_sub_dir = image_dir / "train"
        label_sub_dir = label_dir / "train"        
        
    else:
        return

    if not image_sub_dir.exists():
        image_sub_dir.mkdir()

    if not label_sub_dir.exists():
        label_sub_dir.mkdir()

    # Output Ultralytics YOLO format data with Object Detection
    output_ultralytics_yolo_formats_with_object_detection(label_names, input_dict, image_sub_dir, label_sub_dir, model)

    return


def output_ultralytics_yolo_format_images_and_labels(path_output, label_names, train_dict, valid_dict):

    image_dir = path_output / "images"
    label_dir = path_output / "labels"

    if not image_dir.exists():
        image_dir.mkdir()

    if not label_dir.exists():
        label_dir.mkdir()

    # Build a YOLOv9c model from pretrained weight
    model = YOLO("yolov9c.pt")

    # Display model information (optional)
    model.info()

    # Output validation data
    output_ultralytics_yolo_formats_val_train(label_names, valid_dict, image_dir, label_dir, model, "valid")

    # Output training data    
    # output_ultralytics_yolo_formats_val_train(label_names, train_dict, image_dir, label_dir, model, "train")

    return


def create_ultralytics_yolo_format_from_birds525(args, label_names, train_dict, valid_dict):

    # Create output directory
    path_output = Path(args[2])
    if not path_output.exists():
        path_output.mkdir()

    # Output data.yaml
    output_ultralytics_yolo_format_data_yaml(path_output, label_names)

    # Output images and labels
    output_ultralytics_yolo_format_images_and_labels(path_output, label_names, train_dict, valid_dict)

    return


def main(args):
    if len(args) != 3:
        print('Usage:', args[0], 'path/to/kaggle/birds525/archive/', 'path/to/outputdir/')
        return

    label_names = []
    train_dict = {}
    valid_dict = {}    
    if not check_birds525(args, label_names, train_dict, valid_dict):
        print(args[1], 'does not satisfy the requirement of birds525/archive/')
        return

    print(args[1], 'satisfy the requirement of birds525/archive/')

    create_ultralytics_yolo_format_from_birds525(args, label_names, train_dict, valid_dict)
        
    return


if __name__ == "__main__":
    args = sys.argv
    main(args)
