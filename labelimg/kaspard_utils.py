from kapnet.pointcloud.utils import transform_with_conf
from kapnet.data.datasets import read_sample
import numpy as np
from pathlib import Path

def project_pcd(filePath):
    fpath = Path(filePath)
    con_path = fpath.parents[1] / "conf" / (fpath.stem+".conf")
    img_path = fpath.parents[1] / "image" / (fpath.stem+".png") 
    pcd_path = fpath.parents[1] / "pcd" / (fpath.stem+".pcd")

    sample = read_sample({
        "conf": con_path, "image": img_path, "pcd": pcd_path
    })
    pcd = sample["pcd"]["points"]
    cfg = sample["conf"]
    pcd = transform_with_conf(pcd, cfg)
    pcd = pcd[~np.isnan(pcd[:,0]), :]
    pcd = pcd[pcd[:,-1]>0.1, :]
    pcd = np.clip(pcd, -5, 5)
    return pcd




