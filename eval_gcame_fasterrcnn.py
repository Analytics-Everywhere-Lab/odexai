import numpy as np
import cv2
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms as T
from data.coco.dataloader import coco_gt_loader
from metrics.pp_ebpg import correspond_box, metric
from metrics.del_ins import del_ins
from xai_methods.gcame import GCAME
from xai_methods.tool import get_prediction_fasterrcnn
import os

import warnings
warnings.simplefilter("ignore", FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get pretrained FasterRCNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True, weights="DEFAULT"
)
model.eval().to(device)

# Prepare target layers for G-CAME on FasterRCNN
target_layers = [
    "backbone.body.layer1.0.relu",
    "backbone.body.layer1.1.relu", 
    "backbone.body.layer2.0.relu",
    "backbone.body.layer2.1.relu",
    "backbone.body.layer3.0.relu",
    "backbone.body.layer3.1.relu",
]

gcame = GCAME(model, target_layers, arch="fasterrcnn")

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
    # try:
    # Load and preprocess image
    org_img = cv2.imread(img_path)
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    h, w, c = org_img.shape
    
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(org_img)
    img_np = img_tensor.numpy().transpose(1, 2, 0)
    img_np = (255 * img_np).astype(np.uint8)
    
    file_name = img_path.split("/")[-1]
    name_img = file_name.split(".")[0]

    # Get predictions from FasterRCNN
    with torch.no_grad():
        prediction = model([img_tensor.to(device)])
        boxes, pred_cls, pred_scores = get_prediction_fasterrcnn(prediction, 0.25)
        
        if len(boxes) == 0:
            continue
        
        # Format boxes for compatibility with metrics
        formatted_boxes = []
        for i, box in enumerate(boxes):
            x_min, y_min = box[0]
            x_max, y_max = box[1]
            class_id = pred_cls[i]
            score = pred_scores[i]
            formatted_boxes.append([x_min, y_min, x_max, y_max, score, class_id])
        
        formatted_boxes = np.array(formatted_boxes)
        saliency_maps = np.zeros((len(formatted_boxes), h, w), dtype=np.float32)

    # Generate saliency maps using G-CAME
    model.zero_grad()
    for idx in range(len(formatted_boxes)):
        saliency_map = gcame(img_tensor.to(device), box=formatted_boxes[idx], obj_idx=idx)
        saliency_maps[idx] = saliency_map

    # Calculate metrics
    with torch.no_grad():
        gt_box, idx_correspond = correspond_box(formatted_boxes, info_data[file_name])
        
        if len(idx_correspond) == 0:
            continue
            
        ebpg, pg, count = metric(gt_box, saliency_maps[idx_correspond, :, :])
        ebpg = np.mean(ebpg[count != 0] / count[count != 0])
        mean_ebpg.append(ebpg)
        pg = np.mean(pg[count != 0] / count[count != 0])
        mean_pg.append(pg)
        
        del_auc, count = del_ins(
            model=model,
            img=img_np,
            bbox=formatted_boxes,
            saliency_map=saliency_maps,
            arch="fasterrcnn",
            mode="del",
            step=2000,
        )
        ins_auc, count = del_ins(
            model=model,
            img=img_np,
            bbox=formatted_boxes,
            saliency_map=saliency_maps,
            arch="fasterrcnn",
            mode="ins", 
            step=2000,
        )
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