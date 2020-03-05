import open3d
from kapnet.pointcloud.utils import transform_with_conf
from kapnet.data.datasets import read_sample
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from numba import jit
from collections import defaultdict


def adapt_pcd(pcd):
    pcd[:,0] = (5.0+pcd[:,0])/5.0
    pcd[:,1] = (5.0-pcd[:,1])/5.0
    return pcd

def reverse_adapt_pcd(pcd):
    pcd[:,0] = pcd[:,0]*5.0-5.0
    pcd[:,1] = 5.0-pcd[:,1]*5.0
    return pcd

def project_pcd(filePath):
    fpath = Path(filePath)
    con_path = fpath.parents[1] / "conf" / (fpath.stem+".toml")
    img_path = fpath.parents[1] / "image" / (fpath.stem+".png") 
    pcd_path = fpath.parents[1] / "pcd" / (fpath.stem+".pcd")

    sample = read_sample({
        "conf": con_path, "image": img_path, "pcd": pcd_path
    })
    pcd = sample["pcd"]["points"]
    cfg = sample["conf"]

    zlim = 2.2
    flim = 0.10
    border = 5.0
    pcd = transform_with_conf(pcd, cfg, do_bed_transform=False)
    sample["pcd"] = pcd.copy() #TODO: is copy necessary?
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
