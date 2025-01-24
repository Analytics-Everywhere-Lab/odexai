import torch
from YOLOX.yolox import models
from YOLOX.yolox.utils import postprocess
import YOLOX.yolox.data.data_augment as data_augment
from metrics.del_ins import del_ins
from metrics.pp_ebpg import correspond_box, metric
from xai_methods.dclose import DCLOSE
from xai_methods.tool import visual
from data.coco.dataloader import coco_gt_loader
import cv2
import numpy as np
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get pretrained model and its transform function
model = models.yolox_l(pretrained=True)
model.eval()
transform = data_augment.ValTransform(legacy=False)
output_dir = "xai_methods/output/dclose"
os.makedirs(output_dir, exist_ok=True)

dclose = DCLOSE(arch="yolox", model=model, img_size=(640, 640), n_samples=4000)

mean_del_auc = []
mean_ins_auc = []
mean_ebpg = []
mean_pg = []

# Load all images in the validation folder
img_folder = "data/coco/val2017/"
img_paths = [
    os.path.join(img_folder, img_name)
    for img_name in os.listdir(img_folder)
    if img_name.endswith(".jpg")
]

info_data = coco_gt_loader()

for img_path in tqdm(img_paths):
    # try:
    # Read and transform image
    org_img = cv2.imread(img_path)
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    h, w, c = org_img.shape
    ratio = min(640 / h, 640 / w)
    img, _ = transform(org_img, None, (640, 640))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img_np = img.squeeze().numpy().transpose(1, 2, 0).astype(np.uint8)
    file_name = img_path.split("/")[-1]
    name_img = file_name.split(".")[0]

    obj_idx = 0
    del_auc = np.zeros(80)
    ins_auc = np.zeros(80)
    count = np.zeros(80)

    # Run explanation
    with torch.no_grad():
        out = model(img.to(device))
        box, index = postprocess(
            out, num_classes=80, conf_thre=0.25, nms_thre=0.45, class_agnostic=True
        )
        box = box[0]
        if box is None:
            continue
        rs = dclose(img, box)
    # visual(img_np, rs, box.cpu(), arch="yolox", save_file="test.png")
    np.save(f"{output_dir}/{name_img}.npy", rs)

    with torch.no_grad():
        out = model(img.to(device))
        box, index = postprocess(
            out, num_classes=80, conf_thre=0.25, nms_thre=0.45, class_agnostic=True
        )
        box = box[0]
        if box is None:
            continue
        saliency_map = np.load(f"{output_dir}/{name_img}.npy")
        
        gt_box, idx_correspond = correspond_box(box.cpu().numpy(), info_data[file_name])
        ebpg_img, pg_img, count_img = metric(gt_box, saliency_map[idx_correspond,:,:])
        mean_ebpg.append(np.mean(ebpg_img[count != 0] / count[count != 0]))
        mean_pg.append(np.mean(pg_img[count != 0] / count[count != 0]))
        del_auc, count = del_ins(model, img_np, box, saliency_map, "del", step=2000)
        ins_auc, count = del_ins(model, img_np, box, saliency_map, "ins", step=2000)
        mean_del_auc.append(np.mean(del_auc[count != 0] / count[count != 0]))
        mean_ins_auc.append(np.mean(ins_auc[count != 0] / count[count != 0]))

# Calculate mean metrics over all images
print("Del auc: ", np.mean(mean_del_auc, axis=0))
print("Ins auc: ", np.mean(mean_ins_auc, axis=0))
print("EBPG: ", np.mean(mean_ebpg, axis=0))
print("PG: ", np.mean(mean_pg, axis=0))
