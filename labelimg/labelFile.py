# Copyright (c) 2016 Tzutalin
# Create by TzuTaLin <tzu.ta.lin@gmail.com>

try:
    from PyQt5.QtGui import QImage
except ImportError:
    from PyQt4.QtGui import QImage

from pathlib import Path
import numpy as np
import sys
import toml

from kapnet.data.datasets import read_sample

infix = ".pcd"
suffix = ".toml"
def convertPoints2RotatedBndBox(shape):
    points = shape['points']
    center = shape['center']
    direction = shape['direction']
    cx, cy = center

    w = np.sqrt((points[0][0]-points[1][0]) ** 2 +
        (points[0][1]-points[1][1]) ** 2)

    h = np.sqrt((points[2][0]-points[1][0]) ** 2 +
        (points[2][1]-points[1][1]) ** 2)

    angle = np.degrees(direction+np.pi/2) # invert because image is inverted

    return (round(cx,4),round(cy,4),round(w,4),round(h,4),round(angle,6))


def saveKaspardFormat(filename, shapes, camCfg, default_labels):
    writer = KaspardWriter(filename, camCfg, default_labels=default_labels)

    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        direction = shape["direction"]
        rbbox = convertPoints2RotatedBndBox(shape)
        writer.add_bbox(*rbbox, label)

    if Path(filename).exists():
        oldconfig = KaspardReader.read_conf(filename)
    else:
        oldconfig = None
    writer.save(targetfile=filename, oldConfig=oldconfig)

def readKaspardFormat(filename, default_labels):
    reader = KaspardReader(filename, default_labels=default_labels)
    return reader.getShapes(), reader.getConfig()["camera"]

class KaspardWriter:
    def __init__(self, filename, camCfg, default_labels=None):
        self.filename = filename
        self.roboxlist = []
        self.camCfg = camCfg

    def add_bbox(self, cx, cy, w, h, angle, name, difficult=None):
        robndbox = {'centerX': float(cx), 'centerY': float(cy), 'width': float(w), 'length': float(h),
                    'orientation': float(angle)}
        robndbox['name'] = name
        self.roboxlist.append(robndbox)

    def append_objects(self, config):
        for obj in self.roboxlist:
            config.setdefault(obj["name"], []).append(obj)

    def save(self, targetfile=None, oldConfig=None):
        if oldConfig is None:
            oldConfig = {}
        cam = {**oldConfig["camera"], **self.camCfg}
        config = {"camera": cam}
        self.append_objects(config)

        if targetfile is None:
            filename = self.filename + ".toml"
        else:
            filename = targetfile
        with open(filename, 'w') as out_file:
            toml.dump(config, out_file)


class KaspardReader:
    def __init__(self, filepath, default_labels=None):
        self.shapes = []
        self.filepath = filepath
        self.verified = False
        self.default_labels = default_labels
        self.parse_conf()

    @staticmethod
    def read_conf(configfile):
        config = read_sample({"conf": configfile})["conf"]
        return config

    def parse_conf(self):
        config = KaspardReader.read_conf(self.filepath)
        for skey, section in config.items():
            if skey in self.default_labels:
                for obj in config[skey]:
                    self.addShape(skey, obj)
        self.config = config

    def getShapes(self):
        return self.shapes

    def getConfig(self):
        return self.config

    def addShape(self, label, box):
        cx = box["centerX"]
        cy = box["centerY"]
        w = box["width"]
        h = box["length"]
        angle = np.pi/2-np.radians(box["orientation"]) # Invert for bc vizu is inverted

        p0x, p0y = (cx - w/2, cy - h/2)
        p1x, p1y = (cx + w/2, cy - h/2)
        p2x, p2y = (cx + w/2, cy + h/2)
        p3x, p3y = (cx - w/2, cy + h/2)

        points = [(p0x, p0y), (p1x, p1y), (p2x, p2y), (p3x, p3y)]
        self.shapes.append((label, points, angle, True, None, None, False))
