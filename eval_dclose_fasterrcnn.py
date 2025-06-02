import torch
import torchvision
import os
from data.coco.dataloader import coco_gt_loader
import cv2
import numpy as np
from metrics.del_ins import del_ins
from metrics.pp_ebpg import correspond_box, metric
from torchvision import transforms as T
from tqdm import tqdm
from data.coco.dataloader import coco_gt_loader
from xai_methods.dclose import DCLOSE
from xai_methods.drise import DRISE
from xai_methods.tool import (
    get_prediction_fasterrcnn,
    get_prediction_fasterrcnn_only_boxes,
    get_prediction_fasterrcnn_dclose,
    visual,
)
from skimage.transform import resize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True, weights="DEFAULT"
)
model.eval().to(device)
output_dir = "xai_methods/output/dclose/fasterrcnn"
os.makedirs(output_dir, exist_ok=True)

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
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # (427, 640, 3)
        h, w, c = img.shape
        transform = T.Compose([T.ToTensor()])
        # preprocess image
        inp = transform(img)
        img_np = inp.numpy().transpose(1, 2, 0)
        img_np = (255 * img_np).astype(np.uint8)
        file_name = img_path.split("/")[-1]
        img_name = file_name.split(".")[0]
        with torch.no_grad():
            prediction = model([inp.to(device)])
            dclose_boxes = get_prediction_fasterrcnn_dclose(prediction, 0.8)
            boxes, pred_cls, pred_scores = get_prediction_fasterrcnn(prediction, 0.8)
            for i in range(len(boxes)):
                boxes[i].append(pred_cls[i])
                boxes[i].append(pred_scores[i])
            dclose = DCLOSE(
                arch="faster-rcnn", model=model, img_size=(inp.shape[1:]), n_samples=100
            )
            # Compute the saliency maps for all boxes
            saliency_maps = dclose(inp, dclose_boxes)
            np.save(f"{output_dir}/{img_name}.npy", saliency_maps)

        # Calculate metrics
        saliency_maps = np.load(f"{output_dir}/{img_name}.npy")

        # Convert each box to the desired format and store in a list
        formatted_boxes = []
        for box in boxes:
            x_min, y_min = box[0]
            x_max, y_max = box[1]
            class_id = np.float32(box[2])  # Convert integer to float32 for consistency
            score = box[3]
            # Construct the row with an additional calculated confidence value
            formatted_boxes.append([x_min, y_min, x_max, y_max, score, class_id])

        gt_box, idx_correspond = correspond_box(formatted_boxes, info_data[file_name])
        if len(idx_correspond) == 0:
            continue
        ebpg, pg, count = metric(gt_box, saliency_maps[idx_correspond, :, :])
        mean_ebpg.append(np.mean(ebpg[count != 0] / count[count != 0]))
        mean_pg.append(np.mean(pg[count != 0] / count[count != 0]))
        del_auc, count = del_ins(
            model=model,
            img=img,
            bbox=formatted_boxes,
            saliency_map=saliency_maps,
            arch="fasterrcnn",
            mode="del",
            step=2000,
        )
        ins_auc, count = del_ins(
            model=model,
            img=img,
            bbox=formatted_boxes,
            saliency_map=saliency_maps,
            arch="fasterrcnn",
            mode="ins",
            step=2000,
        )
        mean_del_auc.append(np.mean(del_auc[count != 0] / count[count != 0]))
        mean_ins_auc.append(np.mean(ins_auc[count != 0] / count[count != 0]))
        print(f"Img {img_idx}: del_auc: {del_auc}, ins_auc: {ins_auc}, ebpg: {ebpg}, pg: {pg}")    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        continue
