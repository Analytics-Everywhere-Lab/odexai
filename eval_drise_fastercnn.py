import torch
import torchvision
import os
from data.coco.dataloader import coco_gt_loader
import cv2
import numpy as np
from torchvision import transforms as T
from tqdm import tqdm

from xai_methods.drise import DRISE
from xai_methods.tool import get_prediction_fasterrcnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval().to(device)

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
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # preprocess image
    inp = transform(img)
    # img_np = img.numpy().transpose(1,2,0)
    # img_np = (255 * img_np).astype(np.uint8)
    # name_img = img_path.split('/')[-1].split('.')[0]
    h, w, c = img.shape
    model.to(device)
    prediction = model([inp.to(device)])
    rs = get_prediction_fasterrcnn(prediction, 0.8)
    boxes, pred_cls, pred = rs
    for i in range(len(boxes)):
        boxes[i].append(pred_cls[i])
        boxes[i].append(pred[i])
    drise = DRISE(arch="fasterrcnn", model=model, img_size=(h, w), n_samples=4000, device=device)
    cam = drise(img, boxes[0])