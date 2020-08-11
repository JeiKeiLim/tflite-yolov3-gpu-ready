//
//  YoloJKResult.swift
//  ObjectDetection
//
//  Created by Jongkuk Lim on 2020/06/24.
//  Copyright © 2020 Y Media Labs. All rights reserved.
//

import Foundation
import TensorFlowLite

public struct YoloJKBox {
  var rect: CGRect
  var confidence: Float
  var rawConfidence: [Float]
  var labelIndex: Int
  
  init(rect: CGRect, confidence: Float, labelIndex: Int, rawConfidence: [Float]) {
    self.rect = rect
    self.confidence = confidence
    self.labelIndex = labelIndex
    self.rawConfidence = rawConfidence
  }
}


class YoloJKResult: NSObject {
  struct Size {
    var width: Int
    var height: Int
    var size: Int {
      get {
        return width*height
      }
    }
  }
  var srcSize: Size?
  let modelInputSize: Size
  
  let s01size: Size
  let s02size: Size
  let s03size: Size
  
  let threshold: Float = 0.5
  let iouThreshold: Float = 0.5
  let maxBoxes: Int = 20
  var threshold_sig: Float { // Inverse Sigmoid
    get {
      return log(threshold / (1-threshold))
    }
  }
  let n_classes: Int = 4
  let multiclass_nms = true
  
  let anchors: [[Float]] = [
  [3.89453125, 4.35546875],
   [7.48046875, 6.58203125],
   [3.978515625, 7.80078125],
   [12.6171875, 10.6484375],
   [6.08203125, 13.734375],
   [22.484375, 18.03125],
   [10.4609375, 24.0],
   [40.59375, 34.1875],
   [89.625, 77.375]
   ]
  var n_anchor: Int {
    get {
      return anchors.count / 3
    }
  }
  
  var labels: [String]
  
  private let colorStrideValue = 10
  private let colors = [
    UIColor.red,
    UIColor(displayP3Red: 90.0/255.0, green: 200.0/255.0, blue: 250.0/255.0, alpha: 1.0),
    UIColor.green,
    UIColor.orange,
    UIColor.blue,
    UIColor.purple,
    UIColor.magenta,
    UIColor.yellow,
    UIColor.cyan,
    UIColor.brown
  ]
  
  init(inputWidth: Int, inputHeight: Int, labels: [String]) {
    assert(inputWidth % 32 == 0 && inputHeight % 32 == 0, "Input size must be multiply of 32!")
    
    modelInputSize = Size(width: inputWidth, height: inputHeight)
    s01size = Size(width: inputWidth/32, height: inputHeight/32)
    s02size = Size(width: inputWidth/16, height: inputHeight/16)
    s03size = Size(width: inputWidth/8, height: inputHeight/8)
    
    self.labels = labels
    
    super.init()
  }
  
  func processFeatureMaps(featMap01: Tensor, featMap02: Tensor, featMap03: Tensor) -> ([YoloJKBox], TimeInterval) {
    let startDate = Date()
    
    let feats = self.getTensorOrder(tensors: featMap01, featMap02, featMap03)
    
    let box01 = processFeatureMap(feats[0], anchors: [6, 7, 8].map {self.anchors[$0]})
    let box02 = processFeatureMap(feats[1], anchors: [3, 4, 5].map {self.anchors[$0]})
    let box03 = processFeatureMap(feats[2], anchors: [0, 1, 2].map {self.anchors[$0]})
    
    let boxes: [YoloJKBox] = box01 + box02 + box03
    
    let selectedIdx = multiclass_nms ?
      nonMaxSuppressionMultiClass(numClasses: self.n_classes, boundingBoxes: boxes, scoreThreshold: self.threshold, iouThreshold: iouThreshold, maxPerClass: maxBoxes, maxTotal: maxBoxes*5) :
      nonMaxSuppression(boundingBoxes: boxes, iouThreshold: iouThreshold, maxBoxes: maxBoxes)
    
    let selectedBoxes = selectedIdx.map { boxes[$0] }
    
    let interval: TimeInterval = Date().timeIntervalSince(startDate) * 1000
    
    return (selectedBoxes, interval)
  }
  
  private func processFeatureMap(_ featMap: Tensor, anchors: [[Float]]) -> [YoloJKBox] {
    assert (anchors.count == self.n_anchor)
    
    let feat = FlatArray<Float32>(tensor: featMap)
    
    let confIdx = self.getFeatIdx(offset: 4)
    var bboxes = [YoloJKBox]()
    
    if let boxConfIdx = feat.searchScore(idx: confIdx, threshold: self.threshold_sig) {
      for boxIdx in boxConfIdx {
        let confItem = feat.getArray(boxIdx[0], boxIdx[1], confIdx[boxIdx[2]]-4, length: 5+self.n_classes)
        let box = processBBox(box: confItem, y: boxIdx[0], x: boxIdx[1], featShape: [feat.dimensions[1], feat.dimensions[2]], anchor: anchors[boxIdx[2]])
        
        bboxes.append(box)
      }
    }
    
    return bboxes
  }
  
  func bboxesToInferences(yoloBoxes: [YoloJKBox]) -> [Inference] {
    var resultArray: [Inference] = []
    
    for box in yoloBoxes {
      guard box.confidence >= self.threshold else {
        continue
      }
      
      let newRect = box.rect.applying(CGAffineTransform(scaleX: CGFloat(self.srcSize!.width),
                                                    y: CGFloat(self.srcSize!.height)))
      let colorToAssign = self.colorForClass(withIndex: box.labelIndex + 1)
      
      let inference = Inference(confidence: box.confidence,
                                className: self.labels[box.labelIndex],
                                rect: newRect,
                                displayColor: colorToAssign)
      
      resultArray.append(inference)
    }
    
    resultArray.sort { (first, second) -> Bool in
      return first.confidence  > second.confidence
    }

    return resultArray
  }
  
  private func processBBox(box: [Float32], y: Int, x: Int, featShape: [Int], anchor: [Float]) -> YoloJKBox {
    let x = (box[0].sigmoid + Float(x)) / Float(featShape[1])
    let y = (box[1].sigmoid + Float(y)) / Float(featShape[0])
    let w = exp(box[2]) * anchor[0] / Float(self.modelInputSize.width)
    let h = exp(box[3]) * anchor[1] / Float(self.modelInputSize.height)
    
    let rect = CGRect(x: CGFloat(x - (w/2.0)), y: CGFloat(y - (h/2.0)), width: CGFloat(w), height: CGFloat(h))

    let confidence = box[4].sigmoid
    
    let rawConfidence = box[5...].map { $0.sigmoid * confidence }
    
    let classProbArgSorted = rawConfidence.argSort()
    
    return YoloJKBox(rect: rect, confidence: rawConfidence[classProbArgSorted[0]], labelIndex: classProbArgSorted[0], rawConfidence: rawConfidence)
  }
  
  private func getFeatIdx(offset: Int = 0) -> [Int] {
    let strides = stride(from: 0, to: self.n_anchor, by: 1)
    
    return strides.map { ($0 * (5 + self.n_classes)) + offset }
  }
  
  private func getTensorOrder(tensors: Tensor...) -> [Tensor] {
    let lenghts = tensors.map { $0.data.count }
    var argsorted = Array(tensors.indices)
    argsorted.sort(by: {lenghts[$0] < lenghts[$1]})
    
    return argsorted.map { tensors[$0] }
  }
  
  /// This assigns color for a particular class.
  private func colorForClass(withIndex index: Int) -> UIColor {

    // We have a set of colors and the depending upon a stride, it assigns variations to of the base
    // colors to each object based on its index.
    let baseColor = colors[index % colors.count]

    var colorToAssign = baseColor

    let percentage = CGFloat((colorStrideValue / 2 - index / colors.count) * colorStrideValue)

    if let modifiedColor = baseColor.getModified(byPercentage: percentage) {
      colorToAssign = modifiedColor
    }

    return colorToAssign
  }
  
}

