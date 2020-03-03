from configparser import ConfigParser
import numpy as np

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
        robndbox = {'centerX': cx, 'centerY': cy, 'width': w, 'height': h,
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
    OBJECTS = ["bed", "couch", "armchair", "person"]

    def __init__(self, filepath):
        self.shapes = []
        self.filepath = filepath
        self.verified = False
        self.parse_conf()

    def parse_conf(self):
        confparser = ConfigParser()
        confparser.optionxform = lambda option: option
        config = {s:dict(confparser.items(s)) for s in confparser.sections()}
        print(config)
        for skey, section in config.items():
            for key, item in section.items():
                try:
                    config[skey][key] = float(item)
                except:
                    pass
            print(skey)
            if skey in self.OBJECTS:
                self.addShape(skey, config[skey])
        self.config = config

    def getShapes(self):
        print(self.shapes)
        return self.shapes

    def addShape(self, label, box):
        cx = box["centerX"]
        cy = box["centerY"]
        w = box["width"]
        h = box["height"]
        angle = box["orientation"]

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