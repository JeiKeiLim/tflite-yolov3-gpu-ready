# gonogono - Make drivers relax while waiting traffic signals
Most code is adopted from TensorFlow Lite Object Detection iOS Example Application

**iOS Versions Supported:** iOS 12.0 and above.
**Xcode Version Required:** 10.0 and above

<img src="https://raw.githubusercontent.com/JeiKeiLim/mygifcontainer/master/gonogono/gonogono.gif" />

## Overview
This application detects when drivers stops the car and tells you when you need to go while you are waiting the traffic signal.

## <b>Caution</b>
* This application does not guarantee any safety related issues.
* Please drive safe!

## Prerequisites

* You must have Xcode installed

* You must have a valid Apple Developer ID

* The demo app requires a camera and must be executed on a real iOS device. You can build it and run with the iPhone Simulator but the app raises a camera not found exception.

* You don't need to build the entire TensorFlow library to run the demo, it uses CocoaPods to download the TensorFlow Lite library.

* You'll also need the Xcode command-line tools:
 ```xcode-select --install```
 If this is a new install, you will need to run the Xcode application once to agree to the license before continuing.
## Building the iOS Demo App

1. Install CocoaPods if you don't have it.
```sudo gem install cocoapods```

2. Install the pod to generate the workspace file:
```cd lite/examples/object_detection/ios/```
```pod install```
  If you have installed this pod before and that command doesn't work, try
```pod update```
At the end of this step you should have a file called ```ObjectDetection.xcworkspace```

3. Open **ObjectDetection.xcworkspace** in Xcode.

4. Please change the bundle identifier to a unique identifier and select your development team in **'General->Signing'** before building the application if you are using an iOS device.

5. Build and run the app in Xcode.
You'll have to grant permissions for the app to use the device's camera. Point the camera at various objects and enjoy seeing how the model classifies things!

### Note
_Please do not delete the empty references_ to the .tflite and .txt files after you clone the repo and open the project. These references will be fulfilled once the model and label files are downloaded when the application is built and run for the first time. If you delete the references to them, you can still find that the .tflite and .txt files are downloaded to the Model folder, the next time you build the application. You will have to add the references to these files in the bundle separately in that case.

## Model Used
You can download pre-trained model in below table.

|Model Name|# Params|Model Download|Label Download|
|----------|--------|--------------|--------------|
|MobileNetV1|~9M|[model download](https://drive.google.com/file/d/1U-4Apzc07B85MP_e6M_JZ3IlkT1OiILC/view?usp=sharing)|[label download](https://drive.google.com/file/d/1TfZRrTMj1Yx1b9lB8QpLEM9PKBwrgJPd/view?usp=sharing)|
|MobileNetV1|~2.3M|[model download](https://drive.google.com/file/d/13S6Gi4mACYwX6QGKwlGglzBGgy69XX_-/view?usp=sharing)|[label download](https://drive.google.com/file/d/1TfZRrTMj1Yx1b9lB8QpLEM9PKBwrgJPd/view?usp=sharing)|

Once you have the model file, place the model file into `Model/` directory


## iOS App Details

The app is written entirely in Swift and uses the TensorFlow Lite
[Swift library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/swift)
for performing image classification.

Note: Objective-C developers should use the TensorFlow Lite
[Objective-C library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/objc).
