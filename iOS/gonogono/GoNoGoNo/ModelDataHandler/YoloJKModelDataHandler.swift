// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import CoreImage
import TensorFlowLite
import UIKit
import Accelerate

protocol GoNoGoNoDelegate: class {
  func cameraImageArrived(pixelBuffer: CVPixelBuffer)
  func inferenceArrived(inferences: [Inference], pixelBuffer: CVPixelBuffer)
  func labelInfoArrived(labels: [String])
}

/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct Result {
  let inferenceTime: Double
  let preProcessingTime: Double
  let postProcessingTime: Double
  let inferences: [Inference]
}

/// Information about the YoloV3 model.
enum YoloJK {
  static let modelInfo: FileInfo = (name: "yolo_model", extension: "tflite")
  static let labelsInfo: FileInfo = (name: "yolo_label", extension: "txt")
}

/// Stores one formatted inference.
struct Inference {
  let confidence: Float
  let className: String
  let rect: CGRect
  let displayColor: UIColor
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class YoloJKModelDataHandler: NSObject {
  // MARK: - GONOGONO
  // MARK: - delegate
  weak var gonogonoDelegate: GoNoGoNoDelegate?

  // MARK: - Internal Properties
  /// The current thread count used by the TensorFlow Lite Interpreter.
  let threadCount: Int
  let threadCountLimit = 10

  let threshold: Float = 0.5

  // MARK: Model parameters
  let batchSize = 1
  let inputChannels = 3
  let inputWidth = 256
  let inputHeight = 416

  // MARK: Private properties
  public private(set) var labels: [String] = []
  private var n_classes: Int = 0

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter

  private let bgraPixel = (channels: 4, alphaComponent: 3, lastBgrComponent: 2)
  private let rgbPixelChannels = 3

  private let yoloResult: YoloJKResultDecoder
  public private(set) var isRunning: Bool = false

  // MARK: - Initialization

  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
  /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
  init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int = 1) {
    let modelFilename = modelFileInfo.name

    // Construct the path to the model file.
    guard let modelPath = Bundle.main.path(
      forResource: modelFilename,
      ofType: modelFileInfo.extension
    ) else {
      print("Failed to load the model file with name: \(modelFilename).")
      return nil
    }

    // Specify the options for the `Interpreter`.
//    var coreMLoptions = CoreMLDelegate.Options()
//    coreMLoptions.enabledDevices = .all
//    var delegate: Delegate? = CoreMLDelegate()
    
    var metalOptions = MetalDelegate.Options()
//    metalOptions.allowsPrecisionLoss = true
    metalOptions.waitType = .passive
    
    let delegate = MetalDelegate(options: metalOptions)
    
    self.threadCount = threadCount
    var options = Interpreter.Options()
    options.threadCount = threadCount
    do {
      // Create the `Interpreter`.
        do{
            interpreter = try Interpreter(modelPath: modelPath, options: options, delegates: [delegate])
        } catch let error{
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            print(error.localizedDescription)
            print("Interpreter with Metal GPU Failed. Falling back to CPU")
        }
      // Allocate memory for the model's input `Tensor`s.
      try interpreter.allocateTensors()
    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
      return nil
    }
    
    self.yoloResult = YoloJKResultDecoder(inputWidth: inputWidth, inputHeight: inputHeight, labels: labels)
    
    super.init()

    // Load the classes listed in the labels file.
    self.n_classes = loadLabels(fileInfo: labelsFileInfo)
    self.yoloResult.labels = labels
  }

  /// This class handles all data preprocessing and makes calls to run inference on a given frame
  /// through the `Interpeter`. It then formats the inferences obtained and returns the top N
  /// results for a successful inference.
  func runModel(onFrame pixelBuffer: CVPixelBuffer) -> Result? {
    self.isRunning = true

    let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
    let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
    
    yoloResult.srcSize = YoloJKResultDecoder.Size(width: imageWidth, height: imageHeight)
    
    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
             sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)

    let imageChannels = 4
    assert(imageChannels >= inputChannels)
    // Crops the image to the biggest square in the center and scales it down to model dimensions.
    let scaledSize = CGSize(width: inputWidth, height: inputHeight)
    guard let scaledPixelBuffer = pixelBuffer.resized(to: scaledSize) else {
      return nil
    }

    let preInterval: TimeInterval
    let inferenceInterval: TimeInterval
    
    let featMap01: Tensor
    let featMap02: Tensor
    let featMap03: Tensor

    do {
      let inputTensor = try interpreter.input(at: 0)
      // Remove the alpha component from the image buffer to get the RGB data.
      
      // This takes longer than model running
      let startPre = Date()
      guard let rgbData = rgbDataFromBuffer(
        scaledPixelBuffer,
        byteCount: batchSize * inputWidth * inputHeight * inputChannels,
        isModelQuantized: inputTensor.dataType == .uInt8
      ) else {
        print("Failed to convert the image buffer to RGB data.")
        return nil
      }
      preInterval = Date().timeIntervalSince(startPre) * 1000

      // Copy the RGB data to the input `Tensor`.
      try interpreter.copy(rgbData, toInputAt: 0)

      // Run inference by invoking the `Interpreter`.
      let startDate = Date()
      try interpreter.invoke()
      inferenceInterval = Date().timeIntervalSince(startDate) * 1000

      featMap01 = try interpreter.output(at: 0)
      featMap02 = try interpreter.output(at: 1)
      featMap03 = try interpreter.output(at: 2)
        
    } catch let error {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }
    
    let (yoloBoxes, ppInterval) = yoloResult.processFeatureMaps(featMap01: featMap01, featMap02: featMap02, featMap03: featMap03)
    let resultArray = yoloResult.bboxesToInferences(yoloBoxes: yoloBoxes, sortByBoxSize: true)
    
    gonogonoDelegate?.inferenceArrived(inferences: resultArray, pixelBuffer: scaledPixelBuffer)
    
    isRunning = false
    
//    print(String(format: "%02d (FPS), Run Time : %1fms, Pre-Processing Time: %.1fms, Inference Time: %.1fms, Post Process Time: %.1fms -> %.1fms", Int(1000/(preInterval+inferenceInterval+ppInterval)), preInterval+inferenceInterval+ppInterval, preInterval, inferenceInterval, ppInterval, postInterval))

    // Returns the inference time and inferences
    let result = Result(inferenceTime: inferenceInterval, preProcessingTime: preInterval, postProcessingTime: ppInterval, inferences: resultArray)
    return result
  }

  /// Loads the labels from the labels file and stores them in the `labels` property.
  private func loadLabels(fileInfo: FileInfo) -> Int {
    let filename = fileInfo.name
    let fileExtension = fileInfo.extension
    guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
      fatalError("Labels file not found in bundle. Please add a labels file with name " +
                   "\(filename).\(fileExtension) and try again.")
    }
    do {
      let contents = try String(contentsOf: fileURL, encoding: .utf8)
      labels = contents.components(separatedBy: .newlines)
    } catch {
      fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
                   "valid labels file and try again.")
    }
    return labels.count-1
  }

  /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
  ///
  /// - Parameters
  ///   - buffer: The BGRA pixel buffer to convert to RGB data.
  ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
  ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
  ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
  ///       floating point values).
  /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
  ///     converted.
  private func rgbDataFromBuffer(
    _ buffer: CVPixelBuffer,
    byteCount: Int,
    isModelQuantized: Bool
  ) -> Data? {
    
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer {
      CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
    }
    guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
      return nil
    }
    
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let destinationChannelCount = 3
    let destinationBytesPerRow = destinationChannelCount * width
    
    var sourceBuffer = vImage_Buffer(data: sourceData,
                                     height: vImagePixelCount(height),
                                     width: vImagePixelCount(width),
                                     rowBytes: sourceBytesPerRow)
    
    guard let destinationData = malloc(height * destinationBytesPerRow) else {
      print("Error: out of memory")
      return nil
    }
    
    defer {
      free(destinationData)
    }

    var destinationBuffer = vImage_Buffer(data: destinationData,
                                          height: vImagePixelCount(height),
                                          width: vImagePixelCount(width),
                                          rowBytes: destinationBytesPerRow)
    
    if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA){
      vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    } else if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32ARGB) {
      vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    }

    let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
    if isModelQuantized {
      return byteData
    }

    // Not quantized, convert to floats
    let floats = byteData.map{ Float($0) / 255.0 }
    
    return Data(copyingBufferOf: floats)
  }
}
