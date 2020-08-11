//
//  GoNoGoNo.swift
//  ObjectDetection
//
//  Created by Jongkuk Lim on 2020/06/27.
//  Copyright © 2020 Y Media Labs. All rights reserved.
//

import Foundation
import UIKit

protocol GoNoGoNoStateDelegate: class {
  func go(state: GoNoGoNoState)
  func no(state: GoNoGoNoState)
}

struct GoNoGoNoState {
  private var carMoveState: Bool = true
  var isCarMoving: Bool {
    mutating get {
      if gonogo < 0.3 {
        self.carMoveState = false
      } else if gonogo > 0.7 {
        self.carMoveState = true
      }
      
      return carMoveState
    }
  }
  var frontCarGo: Bool = false
  var trafficLightGo: Bool = false
  var shouldGo: Bool {
    get {
      return frontCarGo || trafficLightGo
    }
  }
  var lastGoTime: Date = Date()
  var gonogo: Double = 1.0
  var boxScore: Double = 999.0
  var tlScore: Double = 0.0
  
  var _dbgBoxSize: Double = 0.0
  var _dbgDiffValue: Double = 0.0
  var _dbgTLRgb: [Double] = [0.0, 0.0, 0.0]
  var _dbgDriveCheckTime: Double = 0.0
  var _dbgBoxProcessTime: Double = 0.0
  var _dbgMsg: String = ""
}

class GoNoGoNo {
  
  // MARK: - Parameters
  let goStopResolutionRatio = 0.05
  let goStopCheckInterval = 1.0
  let mean_img_w_size_ratio = 5.0
  let mean_img_stride_ratio = 1.5
  let boxScoreThreshold = 0.8
  let tlScoreThreshold = 0.1
  
  let window_size = 30
  let diff_store_frame_sec = 5
  var diff_store_frames = 30
  let diff_threshold = 5.0
  
  var labels: [String] = []
  var lastGoStopCheckTime = 0.0
  var driverCenterRatio = 0.5
  
  var lastMeanFrame: [[UInt8]]?
  var diffValues: [Double] = [Double]()
  var boxSizes: [Double] = [Double]()
  var tlRGB: [[Double]] = [[Double]]()
  var lastDiffValue: Double = 1.0
  var lastDiffTime: Date
  var frameArrivalGaps: [Double] = [Double]()
  var lastFrameArrivalTime: Date
  var state: GoNoGoNoState = GoNoGoNoState()
  
  var delegate: GoNoGoNoStateDelegate?
  
  init() {
    lastDiffTime = Date()
    frameArrivalGaps.append(1/30)
    lastFrameArrivalTime = Date()
  }
  
  func judgeGonogo(tlRGB: [Double]? = nil, boxSize: Double? = nil) {
    if self.state.isCarMoving {
      self.boxSizes.popFirst()
      self.tlRGB.popFirst()
    } else {
      if let tlColor = tlRGB {
        self.tlRGB.append(tlColor)
      }
      if let boxsize = boxSize {
        self.boxSizes.append(boxsize)
      }
    }
    self.boxSizes.keepLast(of: window_size)
    self.tlRGB.keepLast(of: window_size)
//    print("IsMoving: \(gonogo), LastDiff: \(self.lastDiffValue)")
    
    var boxScore: Double = 0.0
    var tlScore: Double = 0.0
    
    if self.state.isCarMoving == false{
      if let lastBoxSize = self.boxSizes.last {
        boxScore = lastBoxSize / self.boxSizes.median()
          * (Double(self.window_size)/(Double(self.boxSizes.count)))
        
        self.state.boxScore = boxScore
      }
      if let lastTLRgb = self.tlRGB.last {
        let medianTLRgb = self.tlRGB.median()
        
        tlScore = (medianTLRgb[0]-lastTLRgb[0]) + (lastTLRgb[1]-medianTLRgb[1])
        self.state.tlScore = tlScore
      }
      
      if boxScore < self.boxScoreThreshold {
        self.state.frontCarGo = true
        self.state.lastGoTime = Date()
        self.delegate?.go(state: self.state)
      }
      if tlScore > self.tlScoreThreshold {
        self.state.trafficLightGo = true
        self.state.lastGoTime = Date()
        self.delegate?.go(state: self.state)
      }
    }
    
    let lastRGB = self.tlRGB.last ?? [-1, -1, -1]
    self.state._dbgTLRgb = lastRGB

//    print(String(format: "Box Score: %.2f - %.2f, TL Score: %.2f - (%.2f, %.2f, %.2f), GONOGO? %.2f - %.2f - %d", boxScore, self.boxSizes.last ?? 0.0, tlScore, lastRGB[0], lastRGB[1], lastRGB[2], self.state.gonogo, self.lastDiffValue, self.state.isCarMoving))
//    print("Box Score: \(boxScore) - \(self.boxSizes.last), TL Score: \(tlScore) - \(self.tlRGB.last), GONOGO? \(self.state.gonogo) - \(self.state.isCarMoving)")
  }
  
  func carStopOrGo(pixelBuffer: CVPixelBuffer) -> Bool {
    let sDate = Date()
    let frameArrivalGap = sDate.timeIntervalSince(self.lastFrameArrivalTime)
    self.lastFrameArrivalTime = sDate
    self.frameArrivalGaps.append(frameArrivalGap)
    
    self.diff_store_frames = Int(Double(self.diff_store_frame_sec) / (self.frameArrivalGaps.reduce(0.0, +)/Double(self.frameArrivalGaps.count)))
    self.state._dbgMsg = String(format: "nFrame: %02d", self.diff_store_frames)
    
    self.diffValues.append(self.lastDiffValue > self.diff_threshold ? 1 : 0)
    
    self.diffValues.keepLast(of: self.diff_store_frames)
    self.frameArrivalGaps.keepLast(of: self.diff_store_frames)
    
    self.state.gonogo = self.diffValues.reduce(0.0, +) / Double(self.diffValues.count)
    
    guard sDate.timeIntervalSince(self.lastDiffTime) > self.goStopCheckInterval else {
      return false
    }

    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    
    let targetWidth = CGFloat(Double(width) * goStopResolutionRatio)
    let targetHeight = CGFloat(Double(height) * goStopResolutionRatio)
    
    guard let scaledPixelBuffer = pixelBuffer.resized(to: CGSize(width: targetWidth, height: targetHeight)) else {
      return false
    }
    
    if let grayImage = scaledPixelBuffer.rgbDataFromBuffer(gray: true) {
      let (p0, p1) = grayImage.percentile(p0: 0.1, p1: 0.9) //40 ms
      guard abs(Int(p0) - Int(p1)) > 10 else {
        return false
      }
      let normImage = self.normalizeGrayImage(img: grayImage, p0: p0, p1: p1) //50 ms
      let meanFrame = self.getMeanImageFromGrayImage(image: normImage, dimensions: [Int(targetHeight), Int(targetWidth)]) // 171 ms
      
      if let lastFrame = self.lastMeanFrame {
        if let diffValue = lastFrame.diff(from: meanFrame) {
          self.lastDiffValue = diffValue
          self.state._dbgDiffValue = diffValue
        }
      }
      
      self.lastMeanFrame = meanFrame
      self.lastDiffTime = Date()
      self.state._dbgDriveCheckTime = self.lastDiffTime.timeIntervalSince(sDate)
      
//      print("(\(targetWidth), \(targetHeight) :: p0 = \(p0); p1 = \(p1); diffValue: \(diff)")
//      grayImage.printArrayGray(name: "grayImage")
//      normImage.printArrayGray(name: "normImage")
//      self.lastMeanFrame?.printArray(name: "meanFrame")
      return true
    } else {
      return false
    }
  }
}

extension GoNoGoNo: GoNoGoNoDelegate {
  func cameraImageArrived(pixelBuffer: CVPixelBuffer) {
    let result = self.carStopOrGo(pixelBuffer: pixelBuffer)
    
    if result {
      self.judgeGonogo()
    }
    self.delegate?.no(state: self.state)
  }
  
  func inferenceArrived(inferences: [Inference], pixelBuffer: CVPixelBuffer) {
    guard inferences.count > 0 && self.state.isCarMoving == false else {
      self.judgeGonogo()
      return
    }
    
    let sDate = Date()
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    let img = pixelBuffer.rgbDataFromBuffer(gray: false)
    
//    print("INFERENCE :: w: \(width), h: \(height)")
    var tlColors = Array<Double>(repeating: 0.0, count: 3)
    var tlCnt = 0.0
    var maxBoxSize = 0.0
    for infer in inferences {
      let rect = infer.correctBox(width: CGFloat(width), height: CGFloat(height))
      
      if infer.className == "TrafficLight" {
        if let slicedImg = img?.slice(x: Int(rect.origin.x), y: Int(rect.origin.y), width: Int(rect.width), height: Int(rect.height)) {
//        slicedImg?.printArray(name: "tl_img")
          let tlHist: [Double] = slicedImg.sumColors()
          for i in 0..<tlHist.count {
            tlColors[i] += (tlHist[i] / (Double(slicedImg.count)*Double(slicedImg[0].count)) / 255.0)
          }
          tlCnt += 1
        }
      }
      if ["Car", "Bus", "Truck"].contains(infer.className) {
        let x_center = Double(width) * self.driverCenterRatio
        let box_center_x = (rect.minX + rect.maxX) / 2.0
        let box_ratio = 1 - ((Double(box_center_x) - x_center) / x_center)
        let box_size = Double(rect.width * rect.height).squareRoot() * box_ratio
        maxBoxSize = max(box_size, maxBoxSize)
      }
    }
    
    var tlRGB: [Double]? = nil
    var boxSize: Double? = nil
    if tlCnt > 0 {
      for i in 0..<tlColors.count {
        tlColors[i] /= tlCnt
      }
      tlRGB = tlColors
//      print("R: \(tlColors[0]), G: \(tlColors[0]), B: \(tlColors[0])")
    }
    if maxBoxSize > (Double(width*height).squareRoot() / 5.0) {
      boxSize = maxBoxSize
      self.state._dbgBoxSize = maxBoxSize
    }
    
    self.state._dbgBoxProcessTime = Date().timeIntervalSince(sDate)
    self.judgeGonogo(tlRGB: tlRGB, boxSize: boxSize)
    
  }
  
  func labelInfoArrived(labels: [String]) {
    self.labels = labels
    for label in self.labels {
      print("LABEL :: \(label)")
    }
  }
}


// MARK: - Extension (Image Processing)
extension GoNoGoNo {
  func cvtGrayImage(img: FlatArray<UInt8>, p0: Double = 0.1, p1: Double = 0.9) -> (Array<UInt8>, UInt8, UInt8) {
    var result = [UInt8]()
    for i in 0..<img.dimensions[0] {
      for j in 0..<img.dimensions[1] {
        var grayPixel = 0.0
        for k in 0..<img.dimensions[2] {
          grayPixel += Double(img[i, j, k])
        }
        grayPixel /= Double(img.dimensions[2])
        result.append(UInt8(grayPixel))
      }
    }
    
//    let grayImage = FlatArray(array: result, dimensions: [img.dimensions[0], img.dimensions[1]])
    
    let sortedPixel = result.sorted(by: <)
    let p0Value = sortedPixel[Int(Double(sortedPixel.count) * p0)]
    let p1Value = sortedPixel[Int(Double(sortedPixel.count) * p1)]
    
    return (result, p0Value, p1Value)
  }
  
  func normalizeGrayImage(img: FlatArray<UInt8>, p0: UInt8, p1: UInt8) -> FlatArray<UInt8> {
    let p0 = Float(p0)
    let p1 = Float(p1) - p0
    
    guard p0 < p1 && p1 > 0 else {
      return img
    }
    
    var img = img
    for i in 0..<img.dimensions[0] {
      for j in 0..<img.dimensions[1] {
        img[i, j] = UInt8(min(max(((Float(img[i, j]) - p0) / p1), 0.0), 1.0) * 255)
      }
    }
    
//    return img.compactMap {
//      UInt8(min(max(((Float($0) - p0) / p1), 0.0), 1.0) * 255)
//    }
    return img
  }
  
  func getMeanImageFromGrayImage(image: FlatArray<UInt8>, dimensions: [Int]) -> [[UInt8]] {
//    let image = FlatArray(array: image, dimensions: dimensions)
    
    let win_size_w = Int(Double(image.dimensions[1]) / mean_img_w_size_ratio)
    let win_size_h = Int(Double(image.dimensions[0]) / mean_img_w_size_ratio)
    let stride_w = Int(Double(win_size_w) / mean_img_stride_ratio)
    let stride_h = Int(Double(win_size_h) / mean_img_stride_ratio)
    
    var mean_frame = [[UInt8]]()
    
    for i in stride(from: 0, to: image.dimensions[0], by: stride_h) {
      var row = [UInt8]()
      for j in stride(from: 0, to: image.dimensions[1], by: stride_w) {
        var values = [UInt8]()
        for n in i..<(i+win_size_h) {
          for m in j..<(j+win_size_w) {
            if n >= image.dimensions[0] || m >= image.dimensions[1] {
              continue
            }
            values.append(image[n, m])
          }
        }
        row.append(values.median())
      }
      mean_frame.append(row)
    }
    return mean_frame
  }
  
}
