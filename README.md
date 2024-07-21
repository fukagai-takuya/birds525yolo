### Overview

Create [Ultralytics YOLO format Object Detection](https://docs.ultralytics.com/datasets/detect/) dataset from [BIRDS 525 SPECIES - IMAGE CLASSIFICATION](https://www.kaggle.com/datasets/gpiosenka/100-bird-species) dataset.

The following python script generates Ultralytics YOLO format Object Detection dataset from BIRDS 525 SPECIES - IMAGE CLASSIFICATION dataset.

- create_yolo_dataset_from_birds525.py

The following python script generates Ultralytics YOLO format Object Detection dataset from BIRDS 525 SPECIES - IMAGE CLASSIFICATION dataset. It generates dataset for specified kinds of bird species. 
  
- create_yolo_dataset_from_birds525_limit_bird_species.py

### Usage Examples

- create_yolo_dataset_from_birds525.py

```
python3 ./create_yolo_dataset_from_birds525.py path/to/kaggle/birds525/archive/ path/to/outputdir/
```

- create_yolo_dataset_from_birds525_limit_bird_species.py

```
python3 ./create_yolo_dataset_from_birds525_limit_bird_species.py path/to/kaggle/birds525/archive/ path/to/outputdir/
```

### How to specify target birds species

When using create_yolo_dataset_from_birds525_limit_bird_species.py, specify target birds species by modifying following lines.

- create_yolo_dataset_from_birds525_limit_bird_species.py

```puthon3
def update_train_valid_subdirecotry_dict(subdirectories, labels, subdir_dict):

    ...

    target_bird_species = [
        'BLUE HERON',
        'MALLARD DUCK',
        'EUROPEAN TURTLE DOVE',
        'ROCK DOVE',
    ]

    ...
```
