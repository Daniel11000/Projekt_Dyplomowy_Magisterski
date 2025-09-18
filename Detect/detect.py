import os
import yaml
import torch
import cv2
import numpy as np

from Swin_DETR import DETR, get_spatial_position_embedding, TransformerEncoder, TransformerDecoder
from pycocotools.coco import COCO
import torchvision.transforms.v2 as T

# ---------------------------- CONFIG ---------------------------

MODEL_PATH   = "finetune_my_dataset/best_finetune_my_dataset_weight.pth"
CONFIG_PATH  = "coco_fine_tuning.yaml"

IMAGE_PATH   = "img/photo.jpg"

SAVE_OUTPUT  = False            # <- True żeby zapisać output, False żeby tylko wyświetlić

# ---------------------------------------------------------------

def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)




def build_model_and_load(cfg):
    from collections import OrderedDict

    dp = cfg['dataset_params']
    mp = cfg['model_params']

    model = DETR(
        config=mp,
        num_classes=dp['num_classes'],
        bg_class_idx=dp['bg_class_idx']
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Wczytanie checkpointu
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Jeśli checkpoint zawiera 'model_state_dict', to zostanie użyty
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    # Filtrowanie tylko pasujących warstw
    model_dict = model.state_dict()
    filtered_dict = OrderedDict()
    for k, v in checkpoint.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered_dict[k] = v
        else:
            pass

    # Aktualizacja wag modelu
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)

    return model, device



def make_transform(im_size):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    return T.Compose([
        T.Resize((im_size, im_size)),
        T.ToPureTensor(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

def load_coco_categories(ann_file):
    coco = COCO(ann_file)
    cats = coco.loadCats(coco.getCatIds())
    return {c['id']: c['name'] for c in cats}

def detect_and_draw(model, device, cfg, image_path):
    dp = cfg['dataset_params']
    transform = make_transform(dp['im_size'])
    cat_map   = load_coco_categories(dp['val_ann_file'])


    orig = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1)
    img_tensor = transform(img_t)
    img_tensor = img_tensor.unsqueeze(0).to(device)




    with torch.no_grad():
        out = model(
            img_tensor,
            score_thresh=cfg['train_params']['infer_score_threshold'],
            use_nms=cfg['train_params']['use_nms_infer']
        )
    det = out['detections'][0]
    boxes  = det['boxes'].cpu().numpy()
    labels = det['labels'].cpu().numpy()
    scores = det['scores'].cpu().numpy()

    H, W = orig.shape[:2]
    results = []
    for box, lbl, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = (box * np.array([W, H, W, H])).astype(int)
        name = cat_map.get(int(lbl), str(lbl))
        results.append((name, float(score), (x1,y1,x2,y2)))

        cv2.rectangle(orig, (x1,y1), (x2,y2), (0,255,0), 2)
        text = f"{name}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(orig, (x1, y1-th-4), (x1+tw+4, y1), (0,255,0), -1)
        cv2.putText(orig, text, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

    print(f"Detected {len(results)} objects:")
    for name, score, (x1,y1,x2,y2) in results:
        print(f" - {name} ({score:.2f}) at [{x1},{y1},{x2},{y2}]")

    cv2.imshow("DETR Detection", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- Opcjonalny zapis ---
    if SAVE_OUTPUT:
        base, ext = os.path.splitext(image_path)
        out_path = f"{base}_detected{ext}"
        cv2.imwrite(out_path, orig)
        print(f"Saved output to {out_path}")

if __name__ == "__main__":
    cfg   = load_config(CONFIG_PATH)
    model, device = build_model_and_load(cfg)
    detect_and_draw(model, device, cfg, IMAGE_PATH)
