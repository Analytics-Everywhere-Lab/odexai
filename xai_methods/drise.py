import cv2
import math
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from xai_methods.tool import get_prediction_fasterrcnn_only_boxes
from yolox.utils import postprocess
import YOLOX.yolox.data.data_augment as data_augment
from data.coco.coco_dict import COCO_FASTERRCNN_INDEX_DICT


def bbox_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


class DRISE(object):
    def __init__(
        self,
        arch,
        model,
        img_size=(608, 608),
        grid_size=(16, 16),
        n_samples=1000,
        prob_thresh=0.2,
        batch_size=32,
        seed=0,
        device="cpu",
        **kwargs
    ):
        """
        Parameters:
          - model: The model in nn.Modules() to analyze
          - img_size: The image size in tuple (H, W)
          - grid_size: The grid size in tuple (h, w)
          - n_samples: Number of samples to create
          - prob_thresh: The appearence probability of 1 grid
        """
        self.arch = arch
        self.model = model.eval()
        self.img_size = img_size
        self.grid_size = grid_size
        self.n_samples = n_samples
        self.prob_thresh = prob_thresh
        self.batch_size = batch_size
        self.seed = seed
        self.device = device

    def __call__(self, image, box, obj_idx=None):
        return self.generate_saliency_map(image, box, obj_idx)

    def generate_mask(self):
        """
        Return a mask with shape [H, W]
        """
        image_h, image_w = self.img_size
        grid_h, grid_w = self.grid_size

        # Create cell for mask
        cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
        up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h

        # Create {0, 1} mask
        mask = (
            np.random.uniform(0, 1, size=(grid_h, grid_w)) < self.prob_thresh
        ).astype(np.float32)
        # Up-size to get value in [0, 1]
        mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
        # Randomly crop the mask
        offset_w = np.random.randint(0, cell_w)
        offset_h = np.random.randint(0, cell_h)

        mask = mask[offset_h : offset_h + image_h, offset_w : offset_w + image_w]
        return mask

    def mask_image(self, image, mask):
        """
        Return a masked image with [0, 1] mask
        """
        masked = (
            (image.astype(np.float32) / 255 * np.dstack([mask] * 3)) * 255
        ).astype(np.uint8)
        return masked

    def generate_saliency_map(self, img, box, obj_idx=None):
        if obj_idx is not None:
            box = box[[obj_idx], :]

        np.random.seed(self.seed)
        h, w, c = img.shape
        self.img_size = (h, w)
        num_batches = (self.n_samples + self.batch_size - 1) // self.batch_size

        if self.arch == "yolox":
            transform = data_augment.ValTransform(legacy=False)
            num_objs = box.shape[0]
            target_class = box[:, -1]
            target_box = [
                [(i[0], i[1]), (i[2], i[3])]
                for i in list(box[:, :4].cpu().detach().numpy())
            ]
            saliency_maps = np.zeros((num_objs, h, w), dtype=np.float32)

            # Batch processing for YOLOx
            for batch_idx in tqdm(range(num_batches)):
                current_batch_size = min(
                    self.batch_size, self.n_samples - batch_idx * self.batch_size
                )

                # Generate batch of masks and masked images
                masks = np.zeros((current_batch_size, h, w), dtype=np.float32)
                masked_images = []

                for i in range(current_batch_size):
                    mask = self.generate_mask()
                    masks[i] = mask
                    masked = self.mask_image(img, mask)
                    masked, _ = transform(masked, None, (640, 640))
                    masked_images.append(masked)

                # Stack masked images into batch tensor
                masked_batch = torch.stack(
                    [torch.from_numpy(img) for img in masked_images]
                ).float()

                # Process entire batch at once
                self.model.zero_grad()
                with torch.no_grad():
                    p = self.model(masked_batch.to(self.device))
                    p_box, _ = postprocess(
                        p,
                        num_classes=80,
                        conf_thre=0.25,
                        nms_thre=0.45,
                        class_agnostic=True,
                    )

                # Process each prediction in the batch
                for i in range(current_batch_size):
                    p_box_single = p_box[i] if p_box[i] is not None else None

                    if p_box_single is None:
                        continue

                    pred_class = list(p_box_single[:, -1].cpu().numpy())
                    pred_boxes = [
                        [(j[0], j[1]), (j[2], j[3])]
                        for j in list(p_box_single[:, :4].cpu().detach().numpy())
                    ]
                    pred_score = list(p_box_single[:, 4].cpu().detach().numpy())

                    pred_t = [pred_score.index(x) for x in pred_score if x > 0.5]
                    if len(pred_t) == 0:
                        continue

                    pred_t = pred_t[-1]
                    pred_boxes = pred_boxes[: pred_t + 1]
                    pred_class = pred_class[: pred_t + 1]
                    scores = pred_score[: pred_t + 1]

                    # Process each target object separately
                    for target_idx in range(num_objs):
                        target_cls = target_class[target_idx]
                        target_bbox = target_box[target_idx]

                        best_iou = 0.0
                        best_score = 0.0

                        # Find the best matching prediction for this target object
                        for b in range(len(pred_boxes)):
                            if pred_class[b] != target_cls:
                                continue

                            new_bbox = list(pred_boxes[b][0]) + list(pred_boxes[b][1])
                            target_bbox_flat = list(target_bbox[0]) + list(
                                target_bbox[1]
                            )
                            iou = bbox_iou(new_bbox, target_bbox_flat)

                            if iou > best_iou:
                                best_iou = iou
                                best_score = scores[b]

                        # Update saliency map for this specific target object
                        if best_iou > 0:
                            saliency_contribution = masks[i] * best_iou * best_score
                            saliency_maps[target_idx] += saliency_contribution
        else:
            transform = T.Compose([T.ToTensor()])

            target_class = box[2]
            target_box = list(box[0]) + list(box[1])

            for batch_idx in tqdm(range(num_batches)):
                current_batch_size = min(
                    self.batch_size, self.n_samples - batch_idx * self.batch_size
                )
                masks = np.zeros((current_batch_size, h, w), dtype=np.float32)
                masked_images = np.zeros((current_batch_size, h, w, c), dtype=np.uint8)

                for i in range(current_batch_size):
                    mask = self.generate_mask()
                    masks[i] = mask
                    masked_images[i] = self.mask_image(img, mask)

                masked_images = [
                    transform(Image.fromarray(img)) for img in masked_images
                ]
                masked_images = torch.stack(masked_images).to(self.device)

                with torch.no_grad():
                    predictions = self.model(masked_images)

                for i in range(current_batch_size):
                    prediction = predictions[i]
                    pred_boxes = [
                        [(b[0], b[1]), (b[2], b[3])]
                        for b in prediction["boxes"].cpu().detach().numpy()
                    ]
                    pred_scores = prediction["scores"].cpu().detach().numpy()
                    pred_classes = prediction["labels"].cpu().detach().numpy()
                    pred_classes = [
                        COCO_FASTERRCNN_INDEX_DICT.get(cls, -1) - 1
                        for cls in pred_classes
                        if cls in COCO_FASTERRCNN_INDEX_DICT
                    ]

                    pred_t = np.where(pred_scores > 0.5)[0]
                    if len(pred_t) == 0:
                        continue

                    pred_t = pred_t[-1]
                    pred_boxes = pred_boxes[: pred_t + 1]
                    pred_classes = pred_classes[: pred_t + 1]
                    scores = pred_scores[: pred_t + 1]

                    ious = []
                    all_scores_map = []

                    for b in range(len(pred_boxes)):
                        if pred_classes[b] != target_class:
                            continue
                        else:
                            new_bbox = list(pred_boxes[b][0]) + list(pred_boxes[b][1])
                            iou = bbox_iou(new_bbox, target_box)
                            ious.append(iou)
                            all_scores_map.append(scores[b])

                    if len(ious) == 0:
                        continue

                    t = masks[i] * np.max(ious) * all_scores_map[np.argmax(ious)]
                    saliency_maps += t

        M, m = saliency_maps.max(), saliency_maps.min()
        saliency_maps = (saliency_maps - m) / (M - m)

        return saliency_maps
