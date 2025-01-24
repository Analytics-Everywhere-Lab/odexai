import numpy as np
import torch
import torchvision
import math
import cv2
from YOLOX.yolox.utils import postprocess
import YOLOX.yolox.data.data_augment as data_augment
from scipy import spatial
from tqdm import tqdm
transform = data_augment.ValTransform(legacy=False)


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)  # auc formula


def del_ins(model, img, bbox, saliency_map, mode='del', step=2000, kernel_width=0.25):
    del_ins = np.zeros(80)
    count = np.zeros(80)
    HW = saliency_map.shape[1] * saliency_map.shape[2]
    n_steps = (HW + step - 1) // step
    for idx in range(saliency_map.shape[0]):
        target_cls = bbox[idx][-1]
        if mode == 'del':
            start = img.copy()
            finish = np.zeros_like(start)
        else:
            start = cv2.GaussianBlur(img, (51, 51), 0)
            finish = img.copy()
        salient_order = np.flip(np.argsort(saliency_map[idx].reshape(HW, -1), axis=0), axis=0)
        y = salient_order // img.shape[1]
        x = salient_order - y * img.shape[1]
        scores = np.zeros(n_steps + 1)
        with torch.no_grad():
            for i in tqdm(range(n_steps + 1)):
                temp_ious = []
                temp_score = []
                torch_start = torch.from_numpy(start.transpose(2, 0, 1)).unsqueeze(0).float()
                out = model(torch_start.cuda())
                p_box, index = postprocess(out, num_classes=80, conf_thre=0.25, nms_thre=0.45, class_agnostic=True)
                p_box = p_box[0]
                if p_box is None:
                    scores[i] = 0
                else:
                    for b in p_box:
                        sample_cls = b[-1]
                        sample_box = b[:4]
                        sample_score = b[5:-1]
                        iou = torchvision.ops.box_iou(sample_box[:4].unsqueeze(0),
                                                      bbox[idx][:4].unsqueeze(0)).cpu().item()
                        distances = spatial.distance.cosine(sample_score.cpu(), bbox[idx][5:-1].cpu())
                        weights = math.sqrt(math.exp(-(distances ** 2) / kernel_width ** 2))
                        if target_cls != sample_cls:
                            iou = 0
                            sample_score = torch.tensor(0.)
                        temp_ious.append(iou)
                        s_score = iou * weights
                        temp_score.append(s_score)
                    max_score = temp_score[np.argmax(temp_ious)]
                    scores[i] = max_score
                x_coords = x[step * i:step * (i + 1), :]
                y_coords = y[step * i:step * (i + 1), :]
                start[y_coords, x_coords, :] = finish[y_coords, x_coords, :]
        del_ins[int(target_cls)] += auc(scores)
        count[int(target_cls)] += 1
    return del_ins, count
    