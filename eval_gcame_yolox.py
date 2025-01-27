import numpy as np
import cv2
import torch
from tqdm import tqdm
import YOLOX.yolox.data.data_augment as data_augment
from YOLOX.yolox import models
from YOLOX.yolox.utils import postprocess
from data.coco.dataloader import coco_dataloader
from metrics.pp_ebpg import correspond_box, metric
from metrics.del_ins import del_ins
from xai_methods.dclose import DCLOSE
from xai_methods.gcame import GCAME
from data.coco.dataloader import coco_gt_loader
import os

import warnings
warnings.simplefilter("ignore", FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get pretrained model and its transform function
model = models.yolox_l(pretrained=True)
transform = data_augment.ValTransform(legacy=False)

# Prepare for GCAME output
target_layer = [
    "head.cls_convs.0.0.act",
    "head.cls_convs.0.1.act",
    "head.cls_convs.1.0.act",
    "head.cls_convs.1.1.act",
    "head.cls_convs.2.0.act",
    "head.cls_convs.2.1.act",
]
gcame = GCAME(model, target_layer)

mean_del_auc = []
mean_ins_auc = []
mean_ebpg = []
mean_pg = []

img_folder = "data/coco/val2017/"
img_paths = [
    os.path.join(img_folder, img_name)
    for img_name in os.listdir(img_folder)
    if img_name.endswith(".jpg")
]
info_data = coco_gt_loader()

for img_idx, img_path in tqdm(enumerate(img_paths), total=len(img_paths), desc="Img", leave=False):
    try:
        org_img = cv2.imread(img_path)
        org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
        h, w, c = org_img.shape
        fh, fw = 640, 640
        ratio = min(fh / h, fw / w)
        img, _ = transform(org_img, None, (fh, fw))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img_np = img.squeeze().numpy().transpose(1, 2, 0).astype(np.uint8)
        file_name = img_path.split("/")[-1]
        name_img = file_name.split(".")[0]

        img.requires_grad = False
        model.eval()

        with torch.no_grad():
            out = model(img.to(device))
            boxes, index = postprocess(
                out, num_classes=80, conf_thre=0.25, nms_thre=0.45, class_agnostic=True
            )
            boxes = boxes[0]
            if len(boxes) is None:
                continue
            else:
                saliency_maps = np.zeros((len(boxes), fh, fw), dtype=np.float32)

        model.zero_grad()
        for idx, box in enumerate(boxes):
            saliency_map = gcame(img.to(device), box=box, obj_idx=idx)
            saliency_maps[idx] = saliency_map

        with torch.no_grad():
            gt_box, idx_correspond = correspond_box(boxes.cpu().numpy(), info_data[file_name])
            ebpg, pg, count = metric(gt_box, saliency_maps[idx_correspond, :, :])
            ebpg = np.mean(ebpg[count != 0] / count[count != 0])
            mean_ebpg.append(ebpg)
            pg = np.mean(pg[count != 0] / count[count != 0])
            mean_pg.append(pg)
            del_auc, count = del_ins(
                model=model,
                img=img_np,
                bbox=boxes,
                saliency_map=saliency_maps,
                arch="yolox",
                mode="del",
                step=2000,
            )
            ins_auc, count = del_ins(
                model=model,
                img=img_np,
                bbox=boxes,
                saliency_map=saliency_maps,
                arch="yolox",
                mode="ins",
                step=2000,
            )
            del_auc = np.mean(del_auc[count != 0] / count[count != 0])
            mean_del_auc.append(del_auc)
            ins_auc = np.mean(ins_auc[count != 0] / count[count != 0])
            mean_ins_auc.append(ins_auc)
            print(f"Img {img_idx}: del_auc: {del_auc}, ins_auc: {ins_auc}, ebpg: {ebpg}, pg: {pg}")
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        continue
# Calculate mean metrics over all images
print("Del auc: ", np.mean(mean_del_auc, axis=0))
print("Ins auc: ", np.mean(mean_ins_auc, axis=0))
print("EBPG: ", np.mean(mean_ebpg, axis=0))
print("PG: ", np.mean(mean_pg, axis=0))
