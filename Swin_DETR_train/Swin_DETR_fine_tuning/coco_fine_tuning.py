import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from torchvision.io import read_image
from torchvision import tv_tensors
from pycocotools.coco import COCO
import numpy as np

class CocoDataset(Dataset):
    """
    CocoDataset for training DETR-like models on COCO-format JSONs (including your
    Cityscapes->COCO conversion). This class:
      - builds a mapping from COCO category ids -> contiguous label ids (1..C)
      - exposes `self.num_classes` = C  (number of object classes, excluding BG)
      - returns targets where 'labels' are integers in [1..C]
      - returns targets['boxes'] normalized to [0,1]
    """
    def __init__(self, ann_file, img_dir, im_size=640, split='train'):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.im_size = im_size
        self.split = split

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std  = [0.229, 0.224, 0.225]

        self.transforms = {
            'train': T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomPhotometricDistort(),
                T.Resize((im_size, im_size)),
                T.SanitizeBoundingBoxes(
                    labels_getter=lambda transform_input:
                        (transform_input[1]['labels'],
                         transform_input[1].get('difficult', None))
                ),
                T.ToPureTensor(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]),
            'val': T.Compose([
                T.Resize((im_size, im_size)),
                T.SanitizeBoundingBoxes(
                    labels_getter=lambda transform_input:
                        (transform_input[1]['labels'],
                         transform_input[1].get('difficult', None))
                ),
                T.ToPureTensor(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=imagenet_mean, std=imagenet_std),
            ])
        }

        self.coco_cat_ids = sorted(self.coco.getCatIds())
        self.catid2label = {cid: idx+1 for idx, cid in enumerate(self.coco_cat_ids)}
        self.label2catid = {v:k for k,v in self.catid2label.items()}
        self.num_classes = len(self.coco_cat_ids)  # number of object classes (WITHOUT BG)

        self.img_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs(img_id)[0]



        path = os.path.join(self.img_dir, info['file_name'])



        im = read_image(path)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        raw_boxes = []
        raw_labels = []
        for ann in anns:
            if 'bbox' not in ann or ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            raw_boxes.append([x, y, x + w, y + h])
            mapped = self.catid2label.get(ann['category_id'], None)
            if mapped is None:
                continue
            raw_labels.append(mapped)

        if raw_boxes:
            boxes = torch.as_tensor(raw_boxes, dtype=torch.float32)
            labels = torch.as_tensor(raw_labels, dtype=torch.int64)
            difficult = torch.zeros((labels.shape[0],), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            difficult = torch.zeros((0,), dtype=torch.int64)

        targets = {
            'boxes': tv_tensors.BoundingBoxes(
                boxes, format='XYXY',
                canvas_size=(im.shape[-2], im.shape[-1])
            ),
            'labels': labels,
            'difficult': difficult
        }

        im_tensor, targets = self.transforms[self.split](im, targets)

        boxes_xyxy = targets['boxes']
        try:
            if isinstance(boxes_xyxy, tv_tensors.BoundingBoxes):
                boxes_xyxy = boxes_xyxy.as_tensor()
        except Exception:
            pass

        H, W = im_tensor.shape[-2:]
        wh = torch.tensor([W, H, W, H], dtype=torch.float32, device=im_tensor.device)
        if boxes_xyxy.numel() > 0:
            boxes_norm = boxes_xyxy / wh
        else:
            boxes_norm = boxes_xyxy

        targets['boxes'] = boxes_norm.float()

        return im_tensor, targets, path
