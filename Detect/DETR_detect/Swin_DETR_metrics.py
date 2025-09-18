import os
import sys
import time
import json
from collections import OrderedDict
import importlib.util

import cv2
import torch
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torchvision.transforms.v2 as T

# ---------------------------- CONFIG ---------------------------
# Ścieżka do checkpointu modelu .pth
MODEL_PATH = r"Models/finetune_my_dataset/epoch_500_finetune_my_dataset.pth"

# Ścieżka do pliku z klasą DETR, albo nazwę modułu importowalnego
MODEL_MODULE_PATH = r"Swin_DETR.py"
MODEL_MODULE_NAME = "Swin_DETR"  # używane jeśli MODEL_MODULE_PATH jest pusty

# GROUND-TRUTH z Datasetu
ANNOTATIONS_JSON = r"Dataset/Road_Objects_Dataset/COCO_format/test/new_annotations.coco.json"

# Katalog zawierający obrazy
IMAGES_DIR = r"Dataset/Road_Objects_Dataset/COCO_format/test"

# Gdzie zapisać predykcje (COCO format)
OUTPUT_PRED_JSON = "predictions_500_ep_on_test.json"

# Parametry modelu
NUM_CLASSES = 92         # liczba klas (taka jak podczas trenowania)
BG_CLASS_IDX = 0         # indeks klasy tła
IM_SIZE = 1024           # rozmiar obrazu używany przy inferencji (taki jak podczas trenowania)

INFER_SCORE_THRESHOLD = 0.5
USE_NMS_INFER = True
BATCH_SIZE = 1

CATEGORY_MAP = None
# =====================================================================



def import_module_from_path(path):
    path = os.path.abspath(path)
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def import_model_module():
    if MODEL_MODULE_PATH:
        return import_module_from_path(MODEL_MODULE_PATH)
    else:
        return __import__(MODEL_MODULE_NAME, fromlist=['*'])


def build_model_and_load_from_module(mod, model_path, device):
    if not hasattr(mod, 'DETR'):
        raise RuntimeError("Zaimportowany moduł nie zawiera klasy DETR.")
    DETR_cls = getattr(mod, 'DETR')


    model_params = {
        'im_channels': 3,
        'backbone_channels': 512,
        'd_model': 256,
        'num_queries': 100,
        'freeze_backbone': True,
        'encoder_layers': 4,
        'encoder_attn_heads': 8,
        'decoder_layers': 4,
        'decoder_attn_heads': 8,
        'dropout_prob': 0.1,
        'ff_inner_dim': 2048,
        'cls_cost_weight': 1.0,
        'l1_cost_weight': 5.0,
        'giou_cost_weight': 2.0,
        'bg_class_weight': 0.1,
        'nms_threshold': 0.5,
        'encoder_window_size': 7,
        'encoder_head_dim': 32,
        'encoder_relative_pos_embedding': True,
        'encoder_expand_to_global': True,
        'encoder_use_abs_pos': True,
    }

    model = DETR_cls(config=model_params, num_classes=NUM_CLASSES, bg_class_idx=BG_CLASS_IDX)
    model.to(device)
    model.eval()

    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
        else:
            state = ckpt
    elif isinstance(ckpt, torch.nn.Module):
        return ckpt.to(device)
    else:
        state = ckpt

    if isinstance(state, dict):
        model_state = model.state_dict()
        filtered = OrderedDict()
        for k, v in state.items():
            newk = k[len('module.'):] if k.startswith('module.') else k
            if newk in model_state and model_state[newk].shape == v.shape:
                filtered[newk] = v
        model_state.update(filtered)
        model.load_state_dict(model_state, strict=False)
    else:
        try:
            model.load_state_dict(state)
        except Exception as e:
            raise RuntimeError("Nie udało się załadować checkpointu: " + str(e))
    return model


def make_transform(im_size):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    return T.Compose([
        T.Resize((im_size, im_size)),
        T.ToPureTensor(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

def evaluate_model_on_folder():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    mod = import_model_module()
    print("Zaimportowano moduł:", mod.__name__)
    model = build_model_and_load_from_module(mod, MODEL_PATH, device)
    model.eval()
    print("Model załadowany.")

    coco = COCO(ANNOTATIONS_JSON)
    img_info_list = {img['file_name']: img for img in coco.loadImgs(coco.getImgIds())}

    files = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    files = [f for f in files if f in img_info_list]
    print(f"Znaleziono {len(files)} obrazów pasujących do anotacji w {IMAGES_DIR}.")

    transform = make_transform(IM_SIZE)
    preds = []
    times = []

    score_thresh = INFER_SCORE_THRESHOLD

    for fname in tqdm(files, desc="Inference"):
        img_meta = img_info_list[fname]
        image_id = int(img_meta['id'])
        img_path = os.path.join(IMAGES_DIR, fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print("Nie można wczytać:", img_path)
            continue
        H, W = img_bgr.shape[:2]

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img_rgb).permute(2,0,1)
        inp = transform(img_t).unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            out = model(inp, score_thresh=score_thresh, use_nms=USE_NMS_INFER)
        t1 = time.time()
        times.append(t1 - t0)

        if isinstance(out, dict) and 'detections' in out:
            det_list = out['detections']
            det = det_list[0] if isinstance(det_list, list) and len(det_list)>0 else det_list
        elif isinstance(out, dict) and all(k in out for k in ['boxes','labels','scores']):
            det = out
        else:
            if isinstance(out, dict) and 'pred_logits' in out and 'pred_boxes' in out:
                logits = out['pred_logits'][0].softmax(-1)[..., :-1]
                scores_vals, labels_vals = logits.max(-1)
                boxes_cxcywh = out['pred_boxes'][0]
                cx = boxes_cxcywh[:,0]; cy = boxes_cxcywh[:,1]; bw = boxes_cxcywh[:,2]; bh = boxes_cxcywh[:,3]
                x1 = cx - 0.5*bw; y1 = cy - 0.5*bh; x2 = cx + 0.5*bw; y2 = cy + 0.5*bh
                boxes_xyxy = torch.stack([x1,y1,x2,y2], dim=1)
                keep = scores_vals >= score_thresh
                boxes_xyxy = boxes_xyxy[keep]
                labels_vals = labels_vals[keep]
                scores_vals = scores_vals[keep]
                det = {'boxes': boxes_xyxy, 'labels': labels_vals, 'scores': scores_vals}
            else:
                raise RuntimeError("Nieoczekiwany format outputu modelu. Sprawdź, co zwraca forward.")

        boxes = det['boxes'].cpu().numpy()
        labels = det['labels'].cpu().numpy()
        scores = det['scores'].cpu().numpy()

        for b, lab, sc in zip(boxes, labels, scores):
            if float(sc) < score_thresh:
                continue
            if b.max() <= 1.0 + 1e-6:
                x1 = float(b[0]) * W
                y1 = float(b[1]) * H
                x2 = float(b[2]) * W
                y2 = float(b[3]) * H
            else:
                x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            model_label = int(lab)
            if CATEGORY_MAP is not None:
                category_id = int(CATEGORY_MAP.get(model_label, model_label))
            else:
                category_id = int(model_label)
            preds.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [round(x1,2), round(y1,2), round(w,2), round(h,2)],
                "score": float(sc)
            })

    with open(OUTPUT_PRED_JSON, 'w', encoding='utf-8') as f:
        json.dump(preds, f)
    print("Zapisano predykcje do:", OUTPUT_PRED_JSON)

    coco_gt = COCO(ANNOTATIONS_JSON)
    coco_dt = coco_gt.loadRes(OUTPUT_PRED_JSON)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    times = [t for t in times if t>0]
    if times:
        avg = sum(times)/len(times)
        print(f"Średni czas inference: {avg:.4f}s, FPS ≈ {1.0/avg:.2f}")
    else:
        print("Brak pomiarów czasu.")

if __name__ == "__main__":
    for p in [MODEL_PATH, ANNOTATIONS_JSON, IMAGES_DIR]:
        if not p or not os.path.exists(p):
            print("Błąd: sprawdź ścieżkę:", p)
    evaluate_model_on_folder()
