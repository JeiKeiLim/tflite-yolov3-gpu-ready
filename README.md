# YOLO v3 TensorFlow Lite iOS GPU acceleration

* This projects trains YOLO v3 model with [BDD100k Dataset](https://bair.berkeley.edu/blog/2018/05/30/bdd/)
* And you can convert the trained model into TFLite model that GPU acceleration ready for iOS!

# Training
Please refer to [python](https://github.com/JeiKeiLim/tflite-yolov3-gpu-ready/tree/master/python) directory of this repository.

# iOS Examples
You can find out [iOS](https://github.com/JeiKeiLim/tflite-yolov3-gpu-ready/tree/master/iOS) directory of this repository.

## 1. YOLO object detection example project
Please refer to [iOS/yolojk_iOS](https://github.com/JeiKeiLim/tflite-yolov3-gpu-ready/tree/master/iOS/yolojk_iOS) directory of this repository.

|iPhone X|iPadPro 11"|
|--------|-----------|
|<img src="https://raw.githubusercontent.com/JeiKeiLim/mygifcontainer/master/gonogono/iPhoneX_MBv1_0.5.gif"/>|<img src="https://raw.githubusercontent.com/JeiKeiLim/mygifcontainer/master/gonogono/iPadPro11_MBv1_0.5.gif"/>|

## 2. gonogono - Observing traffic signs and front car instead of the driver
Please refer to [iOS/gonogono](https://github.com/JeiKeiLim/tflite-yolov3-gpu-ready/tree/master/iOS/gonogono) directory of this repository.

<img src="https://raw.githubusercontent.com/JeiKeiLim/mygifcontainer/master/gonogono/gonogono.gif" />

# Performance
|Model Name|# Params|# Label|Device|FPS|
|----------|--------|-------|------|---|
|MobileNetV1|~9M|4|iPhone X|15~20|
|MobileNetV1|~2.3M|4|iPhone X|30~35|
|MobileNetV1|~2.3M|4|iPad Pro 11"|60~70|

# Future update
* ~Make model smaller -> under 5MB~
* Int8 quantization support
* Android example application
* CoreML Support
