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
from xai_methods.drise import DRISE
from xai_methods.tool import get_prediction_fasterrcnn
from skimage.transform import resize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True, weights="DEFAULT"
)
model.eval().to(device)
output_dir = "xai_methods/output/drise/fasterrcnn"
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
transform = T.Compose([T.ToTensor()])


for img_path in tqdm(img_paths):
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # (427, 640, 3)
        h, w, c = img.shape
        # preprocess image
        inp = transform(img)
        img_np = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        file_name = img_path.split("/")[-1]
        img_name = file_name.split(".")[0]
        with torch.no_grad():
            prediction = model([inp.to(device)])
            boxes, pred_cls, pred = get_prediction_fasterrcnn(prediction, 0.8)
            saliency_maps = np.zeros((len(boxes), h, w), dtype=np.float32)
            for i in range(len(boxes)):
                boxes[i].append(pred_cls[i])
                boxes[i].append(pred[i])
            drise = DRISE(
                arch="fasterrcnn",
                model=model,
                img_size=(h, w),
                n_samples=100,
                device=device,
            )
            # Compute the saliency maps for all boxes
            for idx, box in enumerate(boxes):
                saliency_map = drise(img, box)
                saliency_maps[idx] = saliency_map
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

        # Calculate mean metrics over all images
        print("Del auc: ", np.mean(mean_del_auc, axis=0))
        print("Ins auc: ", np.mean(mean_ins_auc, axis=0))
        print("EBPG: ", np.mean(mean_ebpg, axis=0))
        print("PG: ", np.mean(mean_pg, axis=0))
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        continue