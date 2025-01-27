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
# Read and transform image
img_path = "data/coco/val2017/000000000139.jpg"
i=0
for img_path in tqdm(img_paths):
    if i == 2:
        break
    try:
        org_img = cv2.imread(img_path)
        org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
        h, w, c = org_img.shape
        ratio = min(640 / h, 640 / w)
        img, _ = transform(org_img, None, (640, 640))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img_np = img.squeeze().numpy().transpose(1, 2, 0).astype(np.uint8)
        file_name = img_path.split("/")[-1]
        name_img = file_name.split(".")[0]


        img.requires_grad = False
        model.eval()
        obj_idx = 1
        del_auc = np.zeros(80)
        ins_auc = np.zeros(80)
        count = np.zeros(80)    

        with torch.no_grad():
            out = model(img.to(device))
            box, index = postprocess(
                out, num_classes=80, conf_thre=0.25, nms_thre=0.45, class_agnostic=True
            )
            # as there is only 1 input image, box is a tensor with each line representing an object detected in the image -> box[0][1] is the second object detected
            # each line has 7 elements: x1, y1, x2, y2, obj_conf, class_conf, class_ID
            box = box[0]
            if box is None:
                continue

        model.zero_grad()
        saliency_map = gcame(img.to(device), box=box[obj_idx], obj_idx=obj_idx)
        # Expand one dim for saliency map
        if len(saliency_map.shape) == 2:
            saliency_map = np.expand_dims(saliency_map, axis=0)

        with torch.no_grad():
            info_data = coco_gt_loader()
            gt_box, idx_correspond = correspond_box(box.cpu().numpy(), info_data[file_name])
            ebpg, pg, count = metric(
                gt_box[obj_idx].reshape(1, 5), saliency_map[obj_idx:, :]
            )
            mean_ebpg.append(np.mean(ebpg[count != 0] / count[count != 0]))
            mean_pg.append(np.mean(pg[count != 0] / count[count != 0]))
            del_auc, count = del_ins(model, img_np, box, saliency_map, "del", step=2000)
            ins_auc, count = del_ins(model, img_np, box, saliency_map, "ins", step=2000)
            mean_del_auc.append(np.mean(del_auc[count != 0] / count[count != 0]))
            mean_ins_auc.append(np.mean(ins_auc[count != 0] / count[count != 0]))
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        continue
    i+=1
# Calculate mean metrics over all images
print("Del auc: ", np.mean(mean_del_auc, axis=0))
print("Ins auc: ", np.mean(mean_ins_auc, axis=0))
print("EBPG: ", np.mean(mean_ebpg, axis=0))
print("PG: ", np.mean(mean_pg, axis=0))
