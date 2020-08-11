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
import Accelerate
import CoreImage
import Foundation
import TensorFlowLite

// Source From https://github.com/tensorflow/examples/blob/master/lite/examples/posenet/ios/PoseNet/Extensions/TFLiteExtension.swift

// MARK: - Data
extension Data {
  /// Creates a new buffer by copying the buffer pointer of the given array.
  ///
  /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
  ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
  ///     data from the resulting buffer has undefined behavior.
  /// - Parameter array: An array with elements of type `T`.
  init<T>(copyingBufferOf array: [T]) {
    self = array.withUnsafeBufferPointer(Data.init)
  }

  /// Convert a Data instance to Array representation.
  func toArray<T>(type: T.Type) -> [T] where T: AdditiveArithmetic {
    var array = [T](repeating: T.zero, count: self.count / MemoryLayout<T>.stride)
    _ = array.withUnsafeMutableBytes { self.copyBytes(to: $0) }
    return array
  }
}

extension Array {
  /// Creates a new array from the bytes of the given unsafe data.
  ///
  /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
  ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
  ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
  /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
  ///     `MemoryLayout<Element>.stride`.
  /// - Parameter unsafeData: The data containing the bytes to turn into an array.
  init?(unsafeData: Data) {
    guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
    #if swift(>=5.0)
    self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
    #else
    self = unsafeData.withUnsafeBytes {
      .init(UnsafeBufferPointer<Element>(
        start: $0,
        count: unsafeData.count / MemoryLayout<Element>.stride
      ))
    }
    #endif  // swift(>=5.0)
  }
}


// MARK: - Wrappers
/// Struct for handling multidimension `Data` in flat `Array`.
struct FlatArray<Element: AdditiveArithmetic> {
  private var array: [Element]
  var dimensions: [Int]

  init(tensor: Tensor) {
    dimensions = tensor.shape.dimensions
    array = tensor.data.toArray(type: Element.self)
  }

  private func flatIndex(_ index: [Int]) -> Int {
    guard index.count == dimensions.count else {
      fatalError("Invalid index: got \(index.count) index(es) for \(dimensions.count) index(es).")
    }

    var result = 0
    for i in 0..<dimensions.count {
      guard dimensions[i] > index[i] else {
        fatalError("Invalid index: \(index[i]) is bigger than \(dimensions[i])")
      }
      result = dimensions[i] * result + index[i]
    }
    return result
  }

  subscript(_ index: Int...) -> Element {
    get {
      return array[flatIndex(index)]
    }
    set(newValue) {
      array[flatIndex(index)] = newValue
    }
  }
}

// MARK: - FlatArray Custom Extension
extension FlatArray where Element == Float32 {
  func searchScore(idx: [Int], threshold: Float) -> [[Int]]? {
    var result: [[Int]] = []
    
    for i in 0..<self.dimensions[1] {
      for j in 0..<self.dimensions[2] {
        for k in 0..<idx.count {
          let conf = self[0, i, j, idx[k]]
          if conf > threshold {
            result.append([i, j, k])
          }
        }
      }
    }
    
    return result.count > 0 ? result : nil
  }
  
  func getArray(_ y: Int, _ x: Int, _ from: Int, length: Int) -> [Float32]{
    let strides = stride(from: from, to: from+length, by: 1)
    
    return strides.map { self[0, y, x, $0] }
  }
}


// MARK: - Float32
extension Float32 {
  public var sigmoid: Float {
    get {
      return 1.0 / (1.0 + exp(-self))
    }
  }
}

extension Array where Element == Float32 {
  public func argSort(from: Int = 0, to: Int = -1) -> [Int] {
    let to = from > to ? self.count : to
    var argsorted = Array<Int>(self[from..<to].indices)
    argsorted.sort(by: {self[$0] > self[$1]})
    
    return argsorted
  }
}
