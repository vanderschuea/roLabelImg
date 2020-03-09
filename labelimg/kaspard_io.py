import numpy as np
import toml
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from kapnet.data.datasets import read_sample

class KaspardWriter:
    def __init__(self, filename, dbsrc=None, localimg_path=None, default_labels=None):
        self.filename = filename
        self.dbsrc = dbsrc
        self.boxlist = []
        self.roboxlist = []
        self.localimg_path = localimg_path
        self.verified = False
    
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
        config = {"camera": oldConfig["camera"]}
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
