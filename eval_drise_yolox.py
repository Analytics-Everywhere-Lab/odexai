import torch
from YOLOX.yolox import models
from YOLOX.yolox.utils import postprocess
import YOLOX.yolox.data.data_augment as data_augment
from metrics.del_ins import del_ins
from metrics.pp_ebpg import correspond_box, metric
from xai_methods.drise import DRISE
from xai_methods.tool import visual
from data.coco.dataloader import coco_gt_loader
import cv2
import numpy as np
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.yolox_l(pretrained=True)
model.eval()
transform = data_augment.ValTransform(legacy=False)
output_dir = "xai_methods/output/drise/yolox"
os.makedirs(output_dir, exist_ok=True)

drise = DRISE(arch="yolox", model=model, device=device)

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
    # try:
    # Read and transform image
    org_img = cv2.imread(img_path)
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    h, w, c = org_img.shape
    ratio = min(640 / h, 640 / w)
    transformed_img, _ = transform(org_img, None, (640, 640))
    transposed_transformed_img = transformed_img.transpose(1, 2, 0)
    img = torch.from_numpy(transformed_img).unsqueeze(0).float()
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
        rs = drise(transposed_transformed_img, box)
        if box is None:
            continue
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
        ebpg = np.mean(ebpg[count != 0] / count[count != 0])
        mean_ebpg.append(ebpg)
        pg = np.mean(pg[count != 0] / count[count != 0])
        mean_pg.append(pg)
        del_auc, count = del_ins(model, img_np, box, saliency_map, "yolox", "del", step=2000)
        ins_auc, count = del_ins(model, img_np, box, saliency_map, "yolox", "ins", step=2000)
        del_auc = np.mean(del_auc[count != 0] / count[count != 0])
        mean_del_auc.append(del_auc)
        ins_auc = np.mean(ins_auc[count != 0] / count[count != 0])
        mean_ins_auc.append(ins_auc)
        print(f"Img {img_idx}: del_auc: {del_auc}, ins_auc: {ins_auc}, ebpg: {ebpg}, pg: {pg}")
    # except Exception as e:
    #     print(f"Error processing {img_path}: {e}")
    #     continue

# Calculate mean metrics over all images
print("Del auc: ", np.mean(mean_del_auc, axis=0))
print("Ins auc: ", np.mean(mean_ins_auc, axis=0))
print("EBPG: ", np.mean(mean_ebpg, axis=0))
print("PG: ", np.mean(mean_pg, axis=0))