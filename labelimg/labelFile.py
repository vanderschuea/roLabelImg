# Copyright (c) 2016 Tzutalin
# Create by TzuTaLin <tzu.ta.lin@gmail.com>

try:
    from PyQt5.QtGui import QImage
except ImportError:
    from PyQt4.QtGui import QImage

from base64 import b64encode, b64decode
from labelimg.kaspard_io import KaspardWriter, KaspardReader
import os.path
import numpy as np
from pathlib import Path
import sys
import math


class LabelFile(object):
    def __init__(self, default_labels):
        self.shapes = ()
        self.imagePath = None
        self.imageData = None
        self.default_labels = default_labels

    def saveKaspardFormat(self, filename, shapes):
        writer = KaspardWriter(filename, default_labels=self.default_labels)

        for shape in shapes:
            points = shape["points"]
            label = shape["label"]
            direction = shape["direction"]
            rbbox = LabelFile.convertPoints2RotatedBndBox(shape)
            writer.add_bbox(*rbbox, label)
        
        if Path(filename).exists():
            oldconfig = KaspardReader.read_conf(filename)
        else:
            oldconfig = None
        writer.save(targetfile=filename, oldConfig=oldconfig)
  
    # You Hao, 2017/06/121
    @staticmethod
    def convertPoints2RotatedBndBox(shape):
        points = shape['points']
        center = shape['center']
        direction = shape['direction']
        cx, cy = center

        w = math.sqrt((points[0][0]-points[1][0]) ** 2 +
            (points[0][1]-points[1][1]) ** 2)

        h = math.sqrt((points[2][0]-points[1][0]) ** 2 +
            (points[2][1]-points[1][1]) ** 2)

        angle = np.degrees(direction+np.pi/2) # invert because image is inverted

        return (round(cx,4),round(cy,4),round(w,4),round(h,4),round(angle,6))
