from configparser import ConfigParser
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class KaspardWriter:
    def __init__(self, foldername, filename, imgsize, dbsrc=None, localimg_path=None):
        self.foldername = foldername
        self.filename = filename
        self.dbsrc = dbsrc
        self.imgsize = imgsize
        self.boxlist = []
        self.roboxlist = []
        self.localimg_path = localimg_path
        self.verified = False
    
    def add_bbox(self, cx, cy, w, h, angle, name, difficult=None):
        robndbox = {'centerX': cx, 'centerY': cy, 'width': w, 'length': h,
                    'orientation': angle}
        robndbox['name'] = name
        self.roboxlist.append(robndbox)

    def append_objects(self, config):
        for obj in self.roboxlist:
            config[obj["name"]] = obj
    
    def save(self, targetfile=None, oldConfig=None):
        if oldConfig is None:
            oldConfig = {}
        config = oldConfig
        self.append_objects(config)
        confparser = ConfigParser()
        confparser.optionxform = lambda option: option # disable lower-casing
        for section in config:
            confparser[section] = config[section]
        
        if targetfile is None:
            filename = self.filename + ".conf"
        else:
            filename = targetfile
        with open(filename, 'w') as out_file:
            confparser.write(out_file)


class KaspardReader:
    OBJECTS = ["bed", "person", "couch", "armchair", "table", "nightstand"]

    def __init__(self, filepath):
        self.shapes = []
        self.filepath = filepath
        self.verified = False
        self.parse_conf()

    @staticmethod
    def read_conf(configfile):
        confparser = ConfigParser()
        confparser.optionxform = lambda option: option
        confparser.read(configfile)
        config = {s:dict(confparser.items(s)) for s in confparser.sections()}
        for skey, section in config.items():
            for key, item in section.items():
                try:
                    config[skey][key] = float(item)
                except:
                    pass
        return config

    def parse_conf(self):
        confparser = ConfigParser()
        confparser.optionxform = lambda option: option
        confparser.read(self.filepath)
        config = {s:dict(confparser.items(s)) for s in confparser.sections()}
        for skey, section in config.items():
            for key, item in section.items():
                try:
                    config[skey][key] = float(item)
                except:
                    pass
            if skey in self.OBJECTS:
                self.addShape(skey, config[skey])
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
        angle = np.radians(box["orientation"])

        p0x, p0y = self.rotatePoint(cx, cy, cx - w/2, cy - h/2, -angle)
        p1x, p1y = self.rotatePoint(cx, cy, cx + w/2, cy - h/2, -angle)
        p2x, p2y = self.rotatePoint(cx, cy, cx + w/2, cy + h/2, -angle)
        p3x, p3y = self.rotatePoint(cx, cy, cx - w/2, cy + h/2, -angle)

        points = [(p0x, p0y), (p1x, p1y), (p2x, p2y), (p3x, p3y)]
        self.shapes.append((label, points, angle, True, None, None, False))


    def rotatePoint(self, xc,yc, xp,yp, theta):        
        xoff = xp-xc
        yoff = yp-yc

        cosTheta = np.cos(theta)
        sinTheta = np.sin(theta)
        pResx = cosTheta * xoff + sinTheta * yoff
        pResy = - sinTheta * xoff + cosTheta * yoff
        # pRes = (xc + pResx, yc + pResy)
        return xc+pResx, yc+pResy    