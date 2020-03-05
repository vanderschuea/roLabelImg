import open3d
from kapnet.pointcloud.utils import transform_with_conf
from kapnet.data.datasets import read_sample
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from numba import jit


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
    pcd = np.nan_to_num(pcd) # NaN->0<flim => filtered
    sample["pcd"] = pcd.copy() #TODO: is copy necessary?

    selected = (pcd[:,-1]>flim) & (pcd[:,-1]<=zlim) &\
               (pcd[:,0]<=border) & (pcd[:,0]>=-border)& (pcd[:,1]<border)

    img = sample["image"].copy().astype(np.uint8)
    icolor = np.rot90(img,2).reshape(pcd.shape[0])[selected]
    sample["image"] = np.stack([img, img, img], axis=-1)

    pcd = pcd[selected, :]
    pcd = adapt_pcd(pcd)

    cnorm = plt.Normalize(vmin=flim, vmax=zlim)
    cmap = cmx.ScalarMappable(norm=cnorm, cmap=plt.get_cmap("magma"))
    zcolor = np.clip(np.round(cmap.to_rgba(pcd[:,-1])*255), 0,255)
    return pcd, (zcolor, icolor), sample

@jit(nopython=True)
def _segment_img(pcd, D, A, B, ok): # About 10-50x faster with numba
    AB, AD, AP = B-A, D-A, pcd-A
    AB2, AD2 = AB@AB, AD@AD
    APAB, APAD = AP@AB, AP@AD
    ok = ok | ((0<APAB) & (APAB<AB2) & (0<APAD) & (APAD<AD2) )
    return ok

def segment_img(sample, visible_shapes, scale):
    pcd = sample["pcd"]
    pcd = pcd[:,:2]
       
    pcd = adapt_pcd(pcd.copy())  # copy to avoid changing sample["pcd"]
    pcd = pcd*scale

    ok = np.zeros(pcd.shape[0], dtype=np.bool)
    for shape in visible_shapes:
        D = np.array((shape.points[0].x(), shape.points[0].y()))
        A = np.array((shape.points[1].x(), shape.points[1].y()))
        B = np.array((shape.points[2].x(), shape.points[2].y()))
        ok = _segment_img(pcd, D, A, B, ok)  
    img = sample["image"].copy()
    ok = np.rot90(np.reshape(ok, img.shape[:2]), 2)
    if np.sum(ok)>0:
        alpha = 0.7
        im_ok = img[ok]
        img[ok] = alpha*im_ok + (1-alpha)*np.array([255, 0,0])*np.ones_like(im_ok)
        img = img.astype(np.uint8)

    return img, np.transpose(ok.nonzero())
