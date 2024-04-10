import os

from typing import List, Tuple, Dict
import torch
import cv2
import numpy as np

from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt

from torch import Tensor
from datasets.image_reader import build_image_reader

from tqdm import tqdm

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=np.float64)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def visualize_compound(fileinfos, preds, masks, pred_imgs, cfg_vis, cfg_reader):
    """Visualize compound image
    Args:
        fileinfos (_type_): list of file information
        preds (_type_): (N, H, W) prediction
        masks (_type_): (N, H, W) mask
        pred_imgs (_type_): (N, C, H, W) original image
        cfg_vis (_type_): 
        cfg_reader (_type_): 
    """
    vis_dir = cfg_vis.save_dir
    max_score = cfg_vis.get("max_score", None)
    min_score = cfg_vis.get("min_score", None)
    max_score = preds.max() if not max_score else max_score
    min_score = preds.min() if not min_score else min_score

    image_reader = build_image_reader(cfg_reader)
    labels = [int(fileinfo["label"]) for fileinfo in fileinfos]
    clsnames = [fileinfo["clsname"] for fileinfo in fileinfos]
    results = get_classify_results(preds, labels, clsnames, type="max", vis_dir=vis_dir)

    for i, fileinfo in tqdm(enumerate(fileinfos)):
        clsname = fileinfo["clsname"]
        filename = fileinfo["filename"]
        label = fileinfo["label"]
        filedir, filename = os.path.split(filename)
        _, defename = os.path.split(filedir)
        
        result = "miss" if label != results[i] else "correct"
        save_dir = os.path.join(vis_dir, clsname, defename, result)
        os.makedirs(save_dir, exist_ok=True)

        # read image
        h, w = int(fileinfo["height"]), int(fileinfo["width"])
        image = image_reader(fileinfo["filename"])
        pred = preds[i][:, :, None].repeat(3, 2)
        pred = cv2.resize(pred, (w, h))

        # pred imgs
        pred_img = np.transpose(pred_imgs[i],(1,2,0))
        pred_img = np.clip((pred_img * imagenet_std + imagenet_mean) * 255, 0, 255).astype(np.uint8)
        pred_img = cv2.resize(pred_img, (w,h))
        # self normalize just for analysis
        scoremap_self = apply_ad_scoremap(image, normalize(pred))
        # global normalize
        pred = np.clip(pred, min_score, max_score)
        pred = normalize(pred, max_score, min_score)
        scoremap_global = apply_ad_scoremap(image, pred)

        if masks is not None:
            mask = (masks[i] * 255).astype(np.uint8)[:, :, None].repeat(3, 2)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            save_path = os.path.join(save_dir, filename)
            if mask.sum() == 0:
                scoremap = np.vstack([image, pred_img, scoremap_global, scoremap_self])
            else:
                scoremap = np.vstack([image, pred_img, mask, scoremap_global, scoremap_self])
        else:
            scoremap = np.vstack([image, scoremap_global, scoremap_self])

        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, scoremap)

    save_histgrams(preds, labels, clsnames, os.path.join(vis_dir, clsname))
    
def save_histgrams(scores, labels, thresh, auc_score, cls, save_dir: str):
        
    scores_anom = [score for i, score in enumerate(scores) if labels[i] == 1]
    scores_norm = [score for i, score in enumerate(scores) if labels[i] == 0]

    print(len(list(scores_anom)), len(list(scores_norm)))
    
    plt.hist(scores_anom, bins=20, alpha=0.5, color="orange", label='anom')
    plt.hist(scores_norm, bins=20, alpha=0.5, color="blue", label='norm')
    
    plt.axvline(x=thresh, color='r', linestyle='--', label='best threshold')
    plt.title(f"{cls} AUC={auc_score}")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{cls}_hist.png"))
    plt.close() 

def find_best_thresh(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)
    gmeans = (tpr * (1-fpr))**0.5
    ix = np.argmax(gmeans)
    return thresholds[ix], gmeans[ix], auc_score
    

def get_classify_results(preds: Tensor, labels: List[int], clsnames: List[str], type: str = "max", vis_dir: str = "") -> List[int]:
    all_cls = list(set(clsnames))
    thresh_dict = {cls: 0 for cls in all_cls}
    
    if type == "max":
        scores = list(np.max(preds, axis=(1, 2)).astype(np.float64))
        scores_zip = list(zip(scores, labels, clsnames))
        # Caluculate AUC and best threshold for each class.
        for cls in all_cls:
            cls_idx = [i for i, c in enumerate(clsnames) if c == cls]
            cls_preds = preds[cls_idx]
            cls_labels = [labels[i] for i in cls_idx]
            cls_scores = list(np.max(cls_preds, axis=(1, 2)).astype(np.float64))
            
            best_thresh, gmeans, auc_score = find_best_thresh(cls_scores, cls_labels)
            thresh_dict[cls] = best_thresh
            print(f"{cls} AUC={auc_score}")
            print(f"{cls} Best Threshold={best_thresh}, G-Mean={gmeans}")
            
            os.makedirs(os.path.join(vis_dir, cls), exist_ok=True)
            save_histgrams(cls_scores, cls_labels, best_thresh, auc_score, cls, os.path.join(vis_dir, cls))
        
        results = []
        for score, label, clsname in scores_zip:
            if score > thresh_dict[clsname]:
                results.append(1)
            else:
                results.append(0)

        return results
    else:
        raise NotImplementedError(f"{type} is not supported")
    
    
        
    
        
        
    
    


def visualize_single(fileinfos, preds, cfg_vis, cfg_reader):
    vis_dir = cfg_vis.save_dir
    max_score = cfg_vis.get("max_score", None)
    min_score = cfg_vis.get("min_score", None)
    max_score = preds.max() if not max_score else max_score
    min_score = preds.min() if not min_score else min_score

    image_reader = build_image_reader(cfg_reader)

    for i, fileinfo in enumerate(fileinfos):
        clsname = fileinfo["clsname"]
        filename = fileinfo["filename"]
        filedir, filename = os.path.split(filename)
        _, defename = os.path.split(filedir)
        save_dir = os.path.join(vis_dir, clsname, defename)
        os.makedirs(save_dir, exist_ok=True)

        # read image
        h, w = int(fileinfo["height"]), int(fileinfo["width"])
        image = image_reader(fileinfo["filename"])
        pred = preds[i][:, :, None].repeat(3, 2)
        pred = cv2.resize(pred, (w, h))

        # write global normalize image
        pred = np.clip(pred, min_score, max_score)
        pred = normalize(pred, max_score, min_score)
        scoremap_global = apply_ad_scoremap(image, pred)

        save_path = os.path.join(save_dir, filename)
        scoremap_global = cv2.cvtColor(scoremap_global, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, scoremap_global)
