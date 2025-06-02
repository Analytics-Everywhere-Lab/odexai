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
output_dir = "xai_methods/output/dclose/yolox"
os.makedirs(output_dir, exist_ok=True)

dclose = DCLOSE(arch="yolox", model=model, img_size=(640, 640), n_samples=100)

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

for img_idx, img_path in tqdm(enumerate(img_paths), total=len(img_paths), desc="Img", leave=False):
    try:
        # Read and transform image
        org_img = cv2.imread(img_path)
        org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
        h, w, c = org_img.shape
        ratio = min(640 / h, 640 / w)
        img, _ = transform(org_img, None, (640, 640)) # (3, 640, 640)
        tensor_img = torch.from_numpy(img).unsqueeze(0).float()
        img_np = img.transpose(1, 2, 0) # (640, 640, 3)
        file_name = img_path.split("/")[-1]
        img_name = file_name.split(".")[0]

        # Run explanation
        with torch.no_grad():
            out = model(tensor_img.to(device))
            box, index = postprocess(
                out, num_classes=80, conf_thre=0.25, nms_thre=0.45, class_agnostic=True, is_dclose_mode=True
            )
            box = box[0]
            if box is None:
                continue
            saliency_maps = dclose(tensor_img, box)
            # visual(img_np, rs, box.cpu(), arch="yolox", save_file="test.png")
            np.save(f"{output_dir}/{img_name}.npy", saliency_maps)

        with torch.no_grad():
            saliency_maps = np.load(f"{output_dir}/{img_name}.npy")

            gt_box, idx_correspond = correspond_box(
                box.cpu().numpy(), info_data[file_name]
            )
            ebpg, pg, count = metric(
                gt_box, saliency_maps[idx_correspond, :, :]
            )
            mean_ebpg.append(np.mean(ebpg[count != 0] / count[count != 0]))
            mean_pg.append(np.mean(pg[count != 0] / count[count != 0]))
            del_auc, count = del_ins(
                model=model,
                img=img_np,
                bbox=box,
                saliency_map=saliency_maps,
                arch="yolox",
                mode="del",
                is_dclose_mode=True,
                step=2000,
            )
            ins_auc, count = del_ins(
                model=model,
                img=img_np,
                bbox=box,
                saliency_map=saliency_maps,
                arch="yolox",
                mode="ins",
                is_dclose_mode=True,
                step=2000,
            )
            mean_del_auc.append(np.mean(del_auc[count != 0] / count[count != 0]))
            mean_ins_auc.append(np.mean(ins_auc[count != 0] / count[count != 0]))
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        continue

# Calculate mean metrics over all images
print("Del auc: ", np.mean(mean_del_auc, axis=0))
print("Ins auc: ", np.mean(mean_ins_auc, axis=0))
print("EBPG: ", np.mean(mean_ebpg, axis=0))
print("PG: ", np.mean(mean_pg, axis=0))