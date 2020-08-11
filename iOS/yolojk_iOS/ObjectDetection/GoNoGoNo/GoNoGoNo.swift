//
//  GoNoGoNo.swift
//  ObjectDetection
//
//  Created by Jongkuk Lim on 2020/06/27.
//  Copyright © 2020 Y Media Labs. All rights reserved.
//

import Foundation
import UIKit

class GoNoGoNo {
  
}

extension GoNoGoNo: ScaledImageDelegate {
  func didOutput(pixelBuffer: CVPixelBuffer) {
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    
    print("w: \(width), h: \(height)")
  }
}
