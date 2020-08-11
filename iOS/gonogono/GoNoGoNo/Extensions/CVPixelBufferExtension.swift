// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

import Foundation
import Accelerate
import VideoToolbox
import UIKit

// MARK: - CVPixelBuffer
extension CVPixelBuffer {
  /// Returns thumbnail by cropping pixel buffer to biggest square and scaling the cropped image
  /// to model dimensions.
  func resized(to size: CGSize) -> CVPixelBuffer? {

    let imageWidth = CVPixelBufferGetWidth(self)
    let imageHeight = CVPixelBufferGetHeight(self)

    let pixelBufferType = CVPixelBufferGetPixelFormatType(self)

    assert(pixelBufferType == kCVPixelFormatType_32BGRA)

    let inputImageRowBytes = CVPixelBufferGetBytesPerRow(self)
    let imageChannels = 4

    CVPixelBufferLockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0))

    // Finds the biggest square in the pixel buffer and advances rows based on it.
    guard let inputBaseAddress = CVPixelBufferGetBaseAddress(self) else {
      return nil
    }

    // Gets vImage Buffer from input image
    var inputVImageBuffer = vImage_Buffer(data: inputBaseAddress, height: UInt(imageHeight), width: UInt(imageWidth), rowBytes: inputImageRowBytes)

    let scaledImageRowBytes = Int(size.width) * imageChannels
    guard  let scaledImageBytes = malloc(Int(size.height) * scaledImageRowBytes) else {
      return nil
    }

    // Allocates a vImage buffer for scaled image.
    var scaledVImageBuffer = vImage_Buffer(data: scaledImageBytes, height: UInt(size.height), width: UInt(size.width), rowBytes: scaledImageRowBytes)

    // Performs the scale operation on input image buffer and stores it in scaled image buffer.
    let scaleError = vImageScale_ARGB8888(&inputVImageBuffer, &scaledVImageBuffer, nil, vImage_Flags(0))

    CVPixelBufferUnlockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0))

    guard scaleError == kvImageNoError else {
      return nil
    }

    let releaseCallBack: CVPixelBufferReleaseBytesCallback = {mutablePointer, pointer in

      if let pointer = pointer {
        free(UnsafeMutableRawPointer(mutating: pointer))
      }
    }

    var scaledPixelBuffer: CVPixelBuffer?

    // Converts the scaled vImage buffer to CVPixelBuffer
    let conversionStatus = CVPixelBufferCreateWithBytes(nil, Int(size.width), Int(size.height), pixelBufferType, scaledImageBytes, scaledImageRowBytes, releaseCallBack, nil, nil, &scaledPixelBuffer)

    guard conversionStatus == kCVReturnSuccess else {

      free(scaledImageBytes)
      return nil
    }

    return scaledPixelBuffer
  }
  
  func rgbDataFromBuffer(gray: Bool = false) -> FlatArray<UInt8>? {
    CVPixelBufferLockBaseAddress(self, .readOnly)
    defer {
      CVPixelBufferUnlockBaseAddress(self, .readOnly)
    }

    guard let sourceData = CVPixelBufferGetBaseAddress(self) else {
      return nil
    }

    let width = CVPixelBufferGetWidth(self)
    let height = CVPixelBufferGetHeight(self)
    let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(self)
    let destinationChannelCount = 3
    let destinationBytesPerRow = destinationChannelCount * width

    var sourceBuffer = vImage_Buffer(data: sourceData,
                                     height: vImagePixelCount(height),
                                     width: vImagePixelCount(width),
                                     rowBytes: sourceBytesPerRow)
    
    if gray {
      guard let grayData = malloc(height * width) else {
        print("Error: out of memory")
        return nil
      }
      defer {
        free(grayData)
      }
      
      var grayBuffer = vImage_Buffer(data: grayData,
                                     height: vImagePixelCount(height),
                                     width: vImagePixelCount(width),
                                     rowBytes: width)
      
      let redCoefficient: Float = 0.2126
      let greenCoefficient: Float = 0.7152
      let blueCoefficient: Float = 0.0722
      
      let divisor: Int32 = 0x1000
      let fDivisor = Float(divisor)

      var coefficientsMatrix = [
          Int16(redCoefficient * fDivisor),
          Int16(greenCoefficient * fDivisor),
          Int16(blueCoefficient * fDivisor)
      ]
      if (CVPixelBufferGetPixelFormatType(self) == kCVPixelFormatType_32BGRA) {
        (coefficientsMatrix[0], coefficientsMatrix[1]) = (coefficientsMatrix[1], coefficientsMatrix[0])
      }
      
      let preBias: [Int16] = [0, 0, 0, 0]
      let postBias: Int32 = 0
      
      let convResult = vImageMatrixMultiply_ARGB8888ToPlanar8(&sourceBuffer,
                                             &grayBuffer,
                                             &coefficientsMatrix,
                                             divisor,
                                             preBias,
                                             postBias,
                                             vImage_Flags(kvImageNoFlags))
      if convResult != kvImageNoError {
        print("Converting Image Failed! \(convResult.description)")
        return nil
      }
      
      let data = Data(bytes: grayBuffer.data, count: grayBuffer.rowBytes * height)
      return FlatArray(array: data.map { $0 }, dimensions: [height, width])
    } else {
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
      
      if (CVPixelBufferGetPixelFormatType(self) == kCVPixelFormatType_32BGRA){
        vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
      } else if (CVPixelBufferGetPixelFormatType(self) == kCVPixelFormatType_32ARGB) {
        vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
      }
      
      let data = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
      return FlatArray(array: data.map { $0 }, dimensions: [height, width, 3])
    }
  }
}


// MARK: - CGImage
extension CGImage {
  /**
    Creates a new CGImage from a CVPixelBuffer.
    - Note: Not all CVPixelBuffer pixel formats support conversion into a
            CGImage-compatible pixel format.
  */
  public static func create(pixelBuffer: CVPixelBuffer) -> CGImage? {
    var cgImage: CGImage?
    VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &cgImage)
    return cgImage
  }
}

// MARK: - UIImage
extension UIImage {
  /**
    Creates a new UIImage from a CVPixelBuffer.
    - Note: Not all CVPixelBuffer pixel formats support conversion into a
            CGImage-compatible pixel format.
  */
  public convenience init?(pixelBuffer: CVPixelBuffer) {
    if let cgImage = CGImage.create(pixelBuffer: pixelBuffer) {
      self.init(cgImage: cgImage)
    } else {
      return nil
    }
  }
}
