from pycocotools.coco import COCO
import numpy as np
from torchvision import datasets

data_dir = "data/coco/val2017"
ann_file = "data/coco/annotations/instances_val2017.json"


def coco_dataloader():
    """
    Load COCO dataset
    :return: COCO dataset
    """
    dataset = datasets.CocoDetection(root=data_dir, annFile=ann_file)
    return dataset


def coco_gt_loader():
    """
    Extract the class label and bounding box for each image in the MS-COCO validation dataset
    Input: annotations_path (path to file annotations)
    Return: info_data: {"name_img": [[x1, y1, x2, y2], [x1', y1', x2', y2'],...]}
    """
    coco = COCO(ann_file)
    ids = coco.getImgIds()
    info_data = dict()
    class_ids = sorted(coco.getCatIds())

    for id_ in ids:
        im_ann = coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        img_name = im_ann["file_name"]
        anno_ids = coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)
        num_objs = len(objs)
        res = np.zeros((num_objs, 5))
        for ix, obj in enumerate(objs):
            cls = class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
        r = min(640 / height, 640 / width)
        res[:, :4] *= r
        info_data[img_name] = res
    return info_data
