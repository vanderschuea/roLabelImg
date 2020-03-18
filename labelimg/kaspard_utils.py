import open3d
from kapnet.pointcloud.utils import transform_with_conf
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

def adapt_pcd(pcd):
    pcd[:,0] = (5.0+pcd[:,0])/5.0
    pcd[:,1] = (5.0-pcd[:,1])/5.0
    return pcd

def reverse_adapt_pcd(pcd):
    pcd[:,0] = pcd[:,0]*5.0-5.0
    pcd[:,1] = 5.0-pcd[:,1]*5.0
    return pcd

def project_pcd(samplePaths):
    sample = read_sample(samplePaths)

    pcd = sample["pcd"]["points"]
    cfg = sample["conf"]

    zlim = 2.2
    flim = 0.10
    border = 5.0
    pcd = transform_with_conf(pcd, cfg, do_bed_transform=False)
    sample["pcd"] = pcd.copy()
    pcd = np.nan_to_num(pcd) # NaN->0<flim => filtered

    cnorm = plt.Normalize(vmin=flim, vmax=zlim)
    cmap = cmx.ScalarMappable(norm=cnorm, cmap=plt.get_cmap("magma"))
    cmap_img = cmx.ScalarMappable(norm=cnorm, cmap=plt.get_cmap("viridis"))
    zcolor = np.clip(np.round(cmap.to_rgba(pcd[:,-1])*255), 0,255)
    zimg = np.clip(np.round(cmap_img.to_rgba(pcd[:,-1])*255), 0,255)

    selected = (pcd[:,-1]>flim) & (pcd[:,-1]<=zlim) &\
               (pcd[:,0]<=border) & (pcd[:,0]>=-border)& (pcd[:,1]<border)

    img = sample["image"].copy().astype(np.uint8)
    icolor = np.rot90(img,2).reshape(pcd.shape[0])[selected]
    sample["image"] = np.stack([img, img, img], axis=-1)
    sample["zimage"] = np.rot90(zimg[:,:3].reshape(img.shape[:2]+(3,)), 2)

    pcd = pcd[selected, :]
    pcd = adapt_pcd(pcd)

    return pcd, (zcolor[selected,:], icolor), sample

@jit(nopython=True)
def _segment_img(pcd, D, A, B, ok): # About 10-50x faster with numba
    AB, AD, AP = B-A, D-A, pcd-A
    AB2, AD2 = AB@AB, AD@AD
    APAB, APAD = AP@AB, AP@AD
    ok = ok | ((0<APAB) & (APAB<AB2) & (0<APAD) & (APAD<AD2) )
    return ok

def segment_img(sample, visible_shapes, scale, hide_floor):
    pcd = sample["pcd"]
    pcd = pcd_orig = pcd
    pcd = clean_pcd = np.nan_to_num(pcd) # copy to avoid changing sample["pcd"]
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
    imgs = sample["image"].copy(), sample["zimage"].copy()
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


def predict_config(fp, floor_model, bed_model):
    with FileLock(fp.parent / ("."+fp.name+".lock")):
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
        

def predict_server(queue, model_paths):
    networks = init_networks(model_paths)
    while True:
        pcd_file = queue.get()
        predict_config(Path(pcd_file), networks[0], networks[1])

def create_predict_server(model_paths):
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    p = ctx.Process(target=predict_server, args=(queue, model_paths))
    p.start()
    return queue, p

def add_to_predict_queue(_parent, start, end, filelist=None, queue=None):
    if queue is None or filelist is None:
        raise RuntimeError("all arguments should be set !")
    for i in range(start, end+1):
        queue.put(filelist[i])
