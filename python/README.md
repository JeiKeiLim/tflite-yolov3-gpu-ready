# YOLO v3 TensorFlow Lite GPU acceleration model
* This projects trains YOLO v3 model with BDD100k Dataset

# Backbone
* MobileNet v1 - TensorFlow Keras MobileNet v1 does not support TFLite GPU acceleration. I had to re-write the model.

# Dataset
* [BDD100k Dataset](https://bair.berkeley.edu/blog/2018/05/30/bdd/) is used to train the model.
* You can train the model with your own dataset with some code work.

# Training from scratch
## 1. Preparing the dataset
* You can download BDD100k dataset [here](https://bdd-data.berkeley.edu).
* This project only requires 100k Images and labels.
* Once you download the dataset, place dataset at `your_path/images` and `your_path/labels`.
* Directory structure
```
your_path/labels/bdd100k_labels_images_train.json
your_path/labels/bdd100k_labels_images_val.json
your_path/images/100k/test
your_path/images/100k/train
your_path/images/100k/val
```

## 2. Setting python environments
* `python >= 3.7`

* `pip install -r requirements.txt` will install packages you need.

## 3. Configuration files
* Instead of tedious code modification, you can customize dataset, yolo model, and, tflite converting options via configuration json files.

### 3.1. bdd100k.json
``` python
{
  "dataset_type": # dataset name.
  "dataset_path": # dataset root path
  "use_classes": # bdd100k has 10 classes. You select exclusively classes to train here.
  "must_include_classes": # Train images only if the classes specified here exist.
  "rotate_trafficlight": {
    "p": # Probability of rotating traffic lights.
    "ratio_threshold": # Vertical ratio under this threshold will not be rotated.
  },
  "exclude_class_with_attributes": { # You can specify excluding bound box with conditions.
    "traffic light": { # Condition of class name
      "exclude_condition": # Either 'or' or 'and'
      "trafficLightColor": # attribute name: attribute value
    }
  },
  "exclude_class_with_box_ratio": # Exclude bounding boxes with width/height ratio specified here
    "bux": 0.05,
  "input_shape": [ # Resize image shape (width, height)
    256, 416
  ],
  "anchors": # Anchors for YOLO v3. This is auto-generated. 
  "grid_ratio": { # YOLO v3 grid ratio. It is recommended to use default.
    "0": 32,
    "1": 16,
    "2": 8
  }
}
```
### 3.2. yolo_conf.json
* Model configuration

### 3.3. tflite_conf.json
* Converting TFLite model configuration

## 3. Train
### 3.1. KMeans cluster to compute anchors
* `python main.py --mode kmeans` 
* KMeans result will be stored in bdd100k.json dataset configuration file.

### 3.2. Training
* `python main.py --mode train`
* Trained model will automatically saved in `train/model_save_path` in model configuration file

### 3.3. Testing the model with a video
* Video testing - `python main.py --mode video --video video_path.mp4`
* Resize input video - `python main.py --mode video --video video_path.mp4 --video_w width --video_h height`
* Save the result video - `python main.py --mode video --video video_path.mp4 --video-out save_path.mp4`

### 3.4. Converting the trained model to TFLite model
* `python main.py --mode tflite`




