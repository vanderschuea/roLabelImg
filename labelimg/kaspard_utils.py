import open3d
from kapnet.utils.io import read_sample, write_sample
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from numba import jit
from collections import defaultdict
from kapnet.training.predict import init as predict_init, predict_floor, predict_object
from kapnet.annotations.image import generate_sample as image_generate_sample
import multiprocessing as mp
from multiprocessing import Process, Queue
from filelock import FileLock
from tempfile import TemporaryDirectory
from PyQt5.QtGui import QColor


def adapt_pcd(pcd):
    pcd[:,0] = (5.0+pcd[:,0])/5.0
    pcd[:,1] = (5.0-pcd[:,1])/5.0
    return pcd

def reverse_adapt_pcd(pcd):
    pcd[:,0] = pcd[:,0]*5.0-5.0
    pcd[:,1] = 5.0-pcd[:,1]*5.0
    return pcd

class ImgPcd():
    FLIM = 0.10  # Max floor height
    ZFLIM = 2.2   # Max height
    XYLIM = 5.0  # Max distance from camera
    _CNORM = plt.Normalize(vmin=FLIM, vmax=ZFLIM)
    CMAP = cmx.ScalarMappable(norm=_CNORM, cmap=plt.get_cmap("magma"))
    CMAP_IMG = cmx.ScalarMappable(norm=_CNORM, cmap=plt.get_cmap("viridis"))
    def __init__(self, samplePaths):
        sample = read_sample(samplePaths)
        self.pcd = sample["pcd"]["points"]
        self.cfg = sample["conf"]
        self.img = sample["image"].copy().astype(np.uint8)
        self.img3d = np.stack([self.img, self.img, self.img], axis=-1)
        self.rotate_floor(self.cfg)

    def make_cfg(self, **kwargs):
        camera = {**self.cfg["camera"], **kwargs}
        return {**self.cfg, "camera": camera}

    def rotate_floor(self, cfg):
        self.rotated_pcd = fast_twconf(
            self.pcd, cfg#, do_bed_transform=False
        )
        pcd = np.nan_to_num(self.rotated_pcd)

        zcolor = np.clip(np.round(self.CMAP.to_rgba(pcd[:,-1])*255), 0,255)
        zimg = np.clip(np.round(self.CMAP_IMG.to_rgba(pcd[:,-1])*255), 0,255)

        selected = (pcd[:,-1]>self.FLIM) & (pcd[:,-1]<=self.ZFLIM) &\
               (pcd[:,0]<=self.XYLIM) & (pcd[:,0]>=-self.XYLIM)& (pcd[:,1]<self.XYLIM)

        icolor = np.rot90(self.img,2).reshape(pcd.shape[0])[selected]

        self.zimage = np.rot90(zimg[:,:3].reshape(self.img.shape[:2]+(3,)), 2)

        pcd = pcd[selected, :]
        self.pcd2d = adapt_pcd(pcd)
        zcolor = zcolor[selected,:]

        # more efficient caching (QColor call is slow)
        self.zcolor = [QColor(*cx) for cx in zcolor]
        self.icolor = [QColor(cx, cx, cx, 255) for cx in icolor]


    def segment_img(self, visible_shapes, scale, hide_floor):
        pcd = pcd_orig = self.rotated_pcd
        pcd = clean_pcd = np.nan_to_num(pcd)
        pcd = adapt_pcd(pcd[:,:2])
        pcd = pcd*scale

        ok = defaultdict(lambda: np.zeros(pcd.shape[0], dtype=np.bool))
        ok["default"] = np.zeros(pcd.shape[0], dtype=np.bool)
        colors = {}
        for shape in visible_shapes:
            D = np.array((shape.points[0].x(), shape.points[0].y()))
            A = np.array((shape.points[1].x(), shape.points[1].y()))
            B = np.array((shape.points[2].x(), shape.points[2].y()))
            ok[shape.label] = _segment_img(pcd, D, A, B, ok[shape.label])
            colors[shape.label] = shape.segment_color
        imgs = self.img3d.copy(), self.zimage.copy()
        lok = None
        for key, ok_key in ok.items():
            ok_key = (~np.isnan(pcd_orig[:,0])) & ok_key
            if hide_floor:
                ok_key = ok_key & (clean_pcd[:,-1]>0.1)
            ok_key = np.rot90(np.reshape(ok_key, imgs[0].shape[:2]), 2)
            lok = ok_key
            if np.sum(ok_key)>0:
                alpha = 0.4
                for img in imgs:
                    im_ok = img[ok_key]
                    img[ok_key] = alpha*im_ok + (1-alpha)*colors[key]*np.ones_like(im_ok)

        return imgs[0].astype(np.uint8), imgs[1].astype(np.uint8)


@jit(nopython=True)
def _segment_img(pcd, D, A, B, ok): # About 10-50x faster with numba
    AB, AD, AP = B-A, D-A, pcd-A
    AB2, AD2 = AB@AB, AD@AD
    APAB, APAD = AP@AB, AP@AD
    ok = ok | ((0<APAB) & (APAB<AB2) & (0<APAD) & (APAD<AD2) )
    return ok


def init_networks(paths):
    """
        paths: list of paths in string format
    """
    networks = []
    for dir_path in paths:
        dir_path = Path(dir_path)
        networks.append(predict_init(dir_path))
    return networks

def segment_floor(model, pcd):
    return predict_floor(model, {}, pcd)

def segment_bed(model, conf, pcd):
    return predict_object(model, conf, pcd)

def read_pcd(path):
    return read_sample({"pcd": path})["pcd"]

def read_config(path):
    return read_sample({"conf": path})["conf"]

def write_config(path, conf):
    write_sample({"conf": conf}, {"conf": path})

def save_image_from_pcd(path, pcd):
    img = image_generate_sample(None, {"pcd": pcd})
    write_sample({"image": img}, {"image": path})


def predict_config(fp, floor_model, bed_model, temppath):
    with FileLock(temppath / (fp.name+".lock")):
        if (fp.parents[1] / "conf").exists():
            kaspardpath = fp.parents[1] / "conf" / (fp.stem+".toml")
        else:
            kaspardpath = fp.parent / (fp.stem+".toml")

        pcd = read_pcd(fp)

        if not kaspardpath.exists():
            conf = segment_floor(floor_model, pcd)
        else:
            conf = read_config(kaspardpath)
        if "bed" not in conf:
            # Annotate bed
            conf = segment_bed(bed_model, conf, pcd)
            # Save config here
            write_config(kaspardpath, conf)

    return pcd, kaspardpath


def predict_server(queue, model_paths, tempdir):
    networks = init_networks(model_paths)
    while True:
        pcd_file = queue.get()
        try:
            predict_config(Path(pcd_file), networks[0], networks[1], Path(tempdir))
        except:
            pass

def create_predict_server(model_paths):
    tempdir = TemporaryDirectory(prefix="ImgLabel-")
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    p = ctx.Process(target=predict_server, args=(queue, model_paths, tempdir.name))
    p.start()
    return queue, p, tempdir

def add_to_predict_queue(_parent, start, end, filelist=None, queue=None):
    if queue is None or filelist is None:
        raise RuntimeError("all arguments should be set !")
    for i in range(start, end+1):
        queue.put(filelist[i])



@jit(nopython=True)
def rotate3d(vec, rot_mat):
    """Gets rotation matrix from angle and applies it."""
    return (rot_mat@vec.T).T
@jit(nopython=True)
def rotateX(theta, vec):
    rot_mat = np.array([
        [1.,0.,0.,],
        [0.,np.cos(theta), -np.sin(theta)],
        [0.,np.sin(theta),  np.cos(theta)],
    ])
    return rotate3d(vec, rot_mat)
@jit(nopython=True)
def rotateY(theta, vec):
    rot_mat = np.array([
        [np.cos(theta),0., np.sin(theta)],
        [0.,1.,0.,],
        [-np.sin(theta),0.,  np.cos(theta)],
    ])
    return rotate3d(vec, rot_mat)
@jit(nopython=True)
def rotateZ(theta, vec):
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta),0.],
        [np.sin(theta),  np.cos(theta),0.],
        [0.,0.,1.],
    ])
    return rotate3d(vec, rot_mat)

def fast_twconf(pcd, cfg): # about 5~7x faster
    try:
        height = cfg["camera"].get("height", 2.6)
    except KeyError:
        raise ValueError("config doesn't contain camera")
    angles = (
        180-cfg["camera"]["inclination"], cfg["camera"]["lateral_inclination"]
    )
    alpha, beta = (np.radians(a) for a in angles)
    pcd = pcd.astype(np.float)
    pcd = rotateX(alpha, pcd)
    pcd = rotateZ(np.pi, pcd)
    pcd = rotateY(beta,  pcd)
    pcd[:,-1] = pcd[:,-1]+height

    return pcd