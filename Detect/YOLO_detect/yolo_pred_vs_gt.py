
"""
Program porównuje pliki z predykcjami wytrenowanego modelu z Ground Truth (z datasetu) i generuje:
 - metrics.csv (per-class + global)
 - summary.json (szczegóły, per-image, per-class, globalne statystyki)
"""
from pathlib import Path
import json
import csv
from collections import defaultdict
import numpy as np
import cv2
import time
from tqdm import tqdm


# ---------------------------- CONFIG ---------------------------
PRED_DIR = "yolo_predictions"     # <- katalog z wygenerowanymi plikami predykcji
GT_DIR = "Dataset/Road_Objects_Dataset/YOLO11_format/test/labels" # <- katalog z ground truth (Dataset)
IMAGES_DIR = "Dataset/Road_Objects_Dataset/YOLO11_format/test/images" # <- katalog z obrazami
OUTPUT_DIR = "yolo_pred_vs_GT_output" # <- katalog gdzie będą zapisane wyniki


IOU_THRESH = 0.50
CONF_FILTER = 0.0
SAVE_PER_IMAGE = True
VERBOSE = True


# zakresy IoU do obliczenia mAP@[.5:.95]
MAP_IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05).tolist()


EPS = 1e-12

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def xywhn_to_xyxy(xc, yc, w, h, img_w, img_h):
    cx = float(xc) * img_w
    cy = float(yc) * img_h
    w_abs = float(w) * img_w
    h_abs = float(h) * img_h
    x1 = cx - w_abs / 2.0
    y1 = cy - h_abs / 2.0
    x2 = cx + w_abs / 2.0
    y2 = cy + h_abs / 2.0
    return [x1, y1, x2, y2]

def parse_pred_file(txt_path):
    res = []
    if not txt_path.exists():
        return res
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            conf = float(parts[5])
            res.append((cls, xc, yc, w, h, conf))
    return res

def parse_gt_file(txt_path):
    res = []
    if not txt_path.exists():
        return res
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            res.append((cls, xc, yc, w, h))
    return res

def iou_xyxy(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    if interArea == 0.0:
        return 0.0
    boxAArea = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    union = boxAArea + boxBArea - interArea
    if union <= 0.0:
        return 0.0
    return interArea / union

def evaluate_for_threshold(all_gt, all_preds, iou_thresh, num_classes=None):
    class_gt_count = defaultdict(int)
    pred_records = defaultdict(list)
    matched_ious_per_class = defaultdict(list)

    for img_id, gts in all_gt.items():
        for g in gts:
            cls = int(g[0])
            class_gt_count[cls] += 1

    if num_classes is None:
        max_cls = 0
        for cls in list(class_gt_count.keys()):
            if cls > max_cls: max_cls = cls
        for img_preds in all_preds.values():
            for p in img_preds:
                if p['class'] > max_cls: max_cls = p['class']
        num_classes = max_cls + 1

    for img_id, preds in all_preds.items():
        preds_sorted = sorted(preds, key=lambda x: x['conf'], reverse=True)
        gts = all_gt.get(img_id, [])
        used_gt = set()
        for p in preds_sorted:
            pbox = [p['x1'], p['y1'], p['x2'], p['y2']]
            pcls = int(p['class'])
            best_iou = 0.0
            best_gt_idx = -1
            for gi, g in enumerate(gts):
                if gi in used_gt:
                    continue
                gcls, gx1, gy1, gx2, gy2 = g
                if int(gcls) != pcls:
                    continue
                iou = iou_xyxy(pbox, [gx1, gy1, gx2, gy2])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi
            if best_iou >= iou_thresh and best_gt_idx >= 0:
                pred_records[pcls].append((p['conf'], 1))
                matched_ious_per_class[pcls].append(best_iou)
                used_gt.add(best_gt_idx)
            else:
                pred_records[pcls].append((p['conf'], 0))

    results = {}
    aps = []
    for cls in range(num_classes):
        gt_count = class_gt_count.get(cls, 0)
        records = sorted(pred_records.get(cls, []), key=lambda x: x[0], reverse=True)
        if gt_count == 0:
            results[cls] = {'AP': None, 'precision': None, 'recall': None, 'TP': 0, 'FP': len(records), 'FN': 0}
            continue
        if len(records) == 0:
            results[cls] = {'AP': 0.0, 'precision': 0.0, 'recall': 0.0, 'TP': 0, 'FP': 0, 'FN': gt_count}
            aps.append(0.0)
            continue
        confs = np.array([r[0] for r in records], dtype=float)
        tps = np.array([r[1] for r in records], dtype=float)
        fps = 1.0 - tps
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        precisions = tp_cum / (tp_cum + fp_cum + EPS)
        recalls = tp_cum / (gt_count + EPS)
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(len(mpre)-2, -1, -1):
            if mpre[i] < mpre[i+1]:
                mpre[i] = mpre[i+1]
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = 0.0
        for i in idx:
            ap += (mrec[i+1] - mrec[i]) * mpre[i+1]
        TP = int(tp_cum[-1]) if len(tp_cum)>0 else 0
        FP = int(fp_cum[-1]) if len(fp_cum)>0 else 0
        FN = int(gt_count - TP)
        prec_final = float(precisions[-1]) if len(precisions)>0 else 0.0
        rec_final = float(recalls[-1]) if len(recalls)>0 else 0.0
        results[cls] = {'AP': float(ap), 'precision': prec_final, 'recall': rec_final, 'TP': TP, 'FP': FP, 'FN': FN}
        aps.append(ap)

    valid_aps = [v['AP'] for v in results.values() if v['AP'] is not None]
    mAP = float(np.mean(valid_aps)) if len(valid_aps)>0 else 0.0

    all_matched_ious = []
    for li in matched_ious_per_class.values():
        all_matched_ious.extend(li)
    mean_iou = float(np.mean(all_matched_ious)) if len(all_matched_ious)>0 else 0.0

    match_stats = {
        'mean_iou': mean_iou,
        'matched_ious': all_matched_ious,
        'matched_count': len(all_matched_ious)
    }

    return results, mAP, match_stats

def collect_predictions_and_gt(pred_dir, gt_dir, img_dir, conf_filter=0.0):
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    img_dir = Path(img_dir)
    all_preds = {}
    all_gt = {}
    image_sizes = {}

    pred_files = sorted(pred_dir.glob("*.txt"))
    for pred_path in pred_files:
        basename = pred_path.stem
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            cand = img_dir / f"{basename}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            candidates = list(img_dir.rglob(f"{basename}.*"))
            if len(candidates) > 0:
                img_path = candidates[0]
        if img_path is None:
            if VERBOSE:
                print(f"UWAGA: nie znaleziono obrazu dla {basename}.txt -> pomijam")
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            if VERBOSE:
                print(f"UWAGA: nie można wczytać obrazu {img_path} -> pomijam")
            continue
        h, w = img.shape[:2]
        image_sizes[basename] = (w, h)

        preds = parse_pred_file(pred_path)
        preds = [p for p in preds if p[5] >= conf_filter]
        pred_boxes = []
        for (cls, xc, yc, ww, hh, conf) in preds:
            xc = clamp01(xc); yc = clamp01(yc); ww = clamp01(ww); hh = clamp01(hh)
            x1,y1,x2,y2 = xywhn_to_xyxy(xc,yc,ww,hh,w,h)
            pred_boxes.append({'class': int(cls), 'conf': float(conf), 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
        all_preds[basename] = pred_boxes

        gt_path = Path(gt_dir) / f"{basename}.txt"
        gts = parse_gt_file(gt_path)
        gt_boxes = []
        for (cls, xc, yc, ww, hh) in gts:
            xc = clamp01(xc); yc = clamp01(yc); ww = clamp01(ww); hh = clamp01(hh)
            x1,y1,x2,y2 = xywhn_to_xyxy(xc,yc,ww,hh,w,h)
            gt_boxes.append((int(cls), x1, y1, x2, y2))
        all_gt[basename] = gt_boxes

    return all_preds, all_gt, image_sizes




def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_preds, all_gt, image_sizes = collect_predictions_and_gt(PRED_DIR, GT_DIR, IMAGES_DIR, conf_filter=CONF_FILTER)
    if len(all_preds) == 0:
        raise SystemExit("Brak pasujących plików predykcji i obrazów do oceny. Sprawdź PRED_DIR i IMAGES_DIR.")

    if VERBOSE:
        print(f"Przygotowano {len(all_preds)} obrazów do ewaluacji.")

    per_image = {}
    results_at_main, mAP_at_main, match_stats_main = evaluate_for_threshold(all_gt, all_preds, IOU_THRESH)

    total_TP = total_FP = total_FN = 0
    all_matched_ious = []
    for img_id, preds in all_preds.items():
        gts = list(all_gt.get(img_id, []))
        used_gt = set()
        img_TP = 0
        img_FP = 0
        matched_ious_img = []
        for p in sorted(preds, key=lambda x: x['conf'], reverse=True):
            pbox = [p['x1'], p['y1'], p['x2'], p['y2']]
            pcls = int(p['class'])
            best_iou = 0.0
            best_gt_idx = -1
            for gi, g in enumerate(gts):
                if gi in used_gt:
                    continue
                gcls, gx1, gy1, gx2, gy2 = g
                if int(gcls) != pcls:
                    continue
                iou = iou_xyxy(pbox, [gx1, gy1, gx2, gy2])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi
            if best_iou >= IOU_THRESH and best_gt_idx >= 0:
                img_TP += 1
                used_gt.add(best_gt_idx)
                matched_ious_img.append(best_iou)
                all_matched_ious.append(best_iou)
            else:
                img_FP += 1
        img_FN = max(0, len(gts) - len(used_gt))
        total_TP += img_TP
        total_FP += img_FP
        total_FN += img_FN
        if SAVE_PER_IMAGE:
            per_image[img_id] = {'TP': img_TP, 'FP': img_FP, 'FN': img_FN, 'mean_iou': float(np.mean(matched_ious_img)) if len(matched_ious_img)>0 else 0.0}

    overall_precision = float(total_TP / (total_TP + total_FP + EPS))
    overall_recall = float(total_TP / (total_TP + total_FN + EPS))
    overall_f1 = float(2*overall_precision*overall_recall / (overall_precision + overall_recall + EPS))
    overall_mean_iou = float(np.mean(all_matched_ious)) if len(all_matched_ious)>0 else 0.0

    results_at_05, mAP_at_05, _ = evaluate_for_threshold(all_gt, all_preds, 0.5)

    ap_per_thresh = []
    per_thresh_results = {}
    for t in MAP_IOU_THRESHOLDS:
        res_t, mAP_t, _ = evaluate_for_threshold(all_gt, all_preds, float(t))
        ap_per_thresh.append(mAP_t)
        per_thresh_results[f"{t:.2f}"] = {'mAP': mAP_t, 'per_class': res_t}
    mAP_range = float(np.mean(ap_per_thresh)) if len(ap_per_thresh)>0 else 0.0

    per_class_summary = {}
    classes_seen = set(list(results_at_main.keys()))
    for cls in sorted(classes_seen):
        vals = results_at_main.get(cls, {})
        AP = vals.get('AP', None)
        TP = vals.get('TP', 0)
        FP = vals.get('FP', 0)
        FN = vals.get('FN', 0)
        prec = float(TP / (TP + FP + EPS))
        rec = float(TP / (TP + FN + EPS))
        f1 = float(2*prec*rec / (prec + rec + EPS))
        per_class_summary[cls] = {'AP': AP, 'precision': prec, 'recall': rec, 'F1': f1, 'TP': TP, 'FP': FP, 'FN': FN}


    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["class", "AP", "precision", "recall", "F1", "TP", "FP", "FN"])
        for cls, vals in sorted(per_class_summary.items()):
            ap = "" if vals['AP'] is None else f"{vals['AP']:.6f}"
            writer.writerow([cls, ap, f"{vals['precision']:.6f}", f"{vals['recall']:.6f}", f"{vals['F1']:.6f}", vals['TP'], vals['FP'], vals['FN']])
        writer.writerow([])
        writer.writerow(["GLOBAL"])
        writer.writerow(["metric", "value"])
        writer.writerow(["total_images", len(all_preds)])
        writer.writerow(["total_TP", total_TP])
        writer.writerow(["total_FP", total_FP])
        writer.writerow(["total_FN", total_FN])
        writer.writerow(["precision", f"{overall_precision:.6f}"])
        writer.writerow(["recall", f"{overall_recall:.6f}"])
        writer.writerow(["F1", f"{overall_f1:.6f}"])
        writer.writerow(["mean_IoU_on_TP", f"{overall_mean_iou:.6f}"])
        writer.writerow(["mAP@main_iou({:.2f})".format(IOU_THRESH), f"{mAP_at_main:.6f}"])
        writer.writerow(["mAP@0.5", f"{mAP_at_05:.6f}"])
        writer.writerow(["mAP@[.5:.95] (avg)", f"{mAP_range:.6f}"])


    summary = {
        "num_images_evaluated": len(all_preds),
        "iou_threshold_main": IOU_THRESH,
        "conf_filter": CONF_FILTER,
        "total_TP": total_TP,
        "total_FP": total_FP,
        "total_FN": total_FN,
        "precision": overall_precision,
        "recall": overall_recall,
        "F1": overall_f1,
        "mean_IoU_on_TP": overall_mean_iou,
        "mAP_at_main": mAP_at_main,
        "mAP_at_0.5": mAP_at_05,
        "mAP_range_[.5:.95]": mAP_range,
        "per_class": per_class_summary,
        "per_threshold": per_thresh_results
    }
    if SAVE_PER_IMAGE:
        summary['per_image'] = per_image

    json_path = out_dir / "summary.json"
    with open(json_path, "w") as jf:
        json.dump(summary, jf, indent=2)

    if VERBOSE:
        print("Zapisano wyniki:")
        print(f" - CSV: {csv_path}")
        print(f" - JSON: {json_path}")
        print("Podsumowanie (krótko):")
        print(f" Images evaluated: {len(all_preds)}, TP={total_TP}, FP={total_FP}, FN={total_FN}")
        print(f" Precision={overall_precision:.4f}, Recall={overall_recall:.4f}, F1={overall_f1:.4f}")
        print(f" mean IoU on TP = {overall_mean_iou:.4f}")
        print(f" mAP@{IOU_THRESH:.2f} = {mAP_at_main:.6f}")
        print(f" mAP@0.5 = {mAP_at_05:.6f}")
        print(f" mAP@[.5:.95] = {mAP_range:.6f}")


if __name__ == "__main__":
    main()
