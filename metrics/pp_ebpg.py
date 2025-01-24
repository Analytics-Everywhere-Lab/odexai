import math
import numpy as np
import cv2
import torch
import torchvision
from scipy import spatial
from tqdm import trange
from YOLOX.yolox.utils import postprocess
from xai_methods.tool import bbox_iou


def metric(bbox, saliency_map):
    """
    bbox:  type(np.ndarray) - shape:[num_boxes, (4 + 1 + num_classes + 1)] - The ground-truth box matches the prediction box
    saliency_map: type(np.ndarray) - shape:[num_boxes, H, W]
    Return: EBPG/PG metric and number of objects.
    """
    empty = np.zeros_like(saliency_map)
    proportion = np.zeros(80)
    count_idx = np.zeros(80)
    pg = np.zeros(80)
    for idx in range(bbox.shape[0]):
        x1, y1, x2, y2 = bbox[idx][:4]
        max_point = np.where(saliency_map[idx] == np.max(saliency_map[idx]))
        cls = int(bbox[idx][-1])
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x1 <= max_point[1][0] <= x2 and y1 <= max_point[0][0] <= y2:
            pg[cls] += 1
        empty[idx][y1:y2, x1:x2] = 1
        mask_bbox = saliency_map[idx] * empty[idx]
        energy_bbox = mask_bbox.sum()
        energy_whole = saliency_map[idx].sum()
        if energy_whole == 0:
            proportion[cls] += 0
            count_idx[cls] += 1
        else:
            proportion[cls] += energy_bbox / energy_whole
            count_idx[cls] += 1
    return proportion, pg, count_idx


def correspond_box(predictbox, groundtruthboxes):
    """
    predictbox: type(np.ndarray) - shape:[num_boxes, (4 + 1 + num_classes + 1)] - Predicted bounding boxes
    groundtruthboxes: type(np.ndarray) - shape:[num_boxes, (4 + 1 + num_classes + 1)] - Ground-truth bounding boxes
    Return: The ground-truth box matches the prediction box and the corresponding index of the prediction box.
    """
    gt_boxs = []
    det = np.zeros(len(groundtruthboxes))
    idx_predictbox = []
    for d in range(len(predictbox)):
        iouMax = 0
        for i in range(len(groundtruthboxes)):
            if predictbox[d][-1] != groundtruthboxes[i][-1]:
                continue
            iou = bbox_iou(predictbox[d][:4], groundtruthboxes[i][:4])
            if iou > iouMax:
                iouMax = iou
                index = i
        if iouMax > 0.5:
            if det[index] == 0:
                det[index] == 1
                gt_boxs.append(groundtruthboxes[index])
                idx_predictbox.append(d)
    return np.array(gt_boxs), idx_predictbox
