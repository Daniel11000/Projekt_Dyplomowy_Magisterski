
"""
Program generuje pliki z predykcjami bounding boxów
Program tylko generuje pliki predykcji (bez porównywania z GT).
"""
import os
import json
import math
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# ---------------------------- CONFIG ---------------------------
GT_JSON_PATH = r"Dataset/Road_Objects_Dataset/COCO_format/test/new_annotations.coco.json" # <-- ścieżka do ground-truth z Datasetu
PRED_JSON_PATH = r"Predictions/Se_DETR_pred/predictions_500_ep.json"

IMAGES_DIR = r"D:\ścieżka\do\images" # katalog z obrazami (używane przy eksportach FP/FN)

INPUT_BBOX_FORMAT = "xywh"
OUTPUT_DIR = r".\evaluation_results_test" # <-- katalog wyjściowy

PRED_TIMES_JSON = None            # opcjonalnie: ścieżka do pliku z inference times [{image_id:.., time_seconds:..}, ...]
TOTAL_INFERENCE_TIME_SECONDS = None  # alternatywnie możesz podać łączny czas inferencji (float)
IOU_FOR_MICRO_EVAL = 0.5
SCORE_THRESHOLD_FOR_POINT_METRICS = 0.5
MAX_DETS = [1, 10, 100]
EXPORT_TOP_N_FP = 20
EXPORT_TOP_N_FN = 20
EXPORT_FP_FN_IMAGES = False  # True aby zapisać obrazy z narysowanymi bboxami



try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception as e:
    raise ImportError(
        "Nie można zaimportować pycocotools. Zainstaluj:\n"
        "pip install pycocotools numpy pandas matplotlib pillow tqdm\n"
        "Na Windowsie czasem: pip install pycocotools-windows\n\n"
        f"Szczegóły błędu: {e}"
    )

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)

def load_json(path):
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def xyxy_to_xywh(b):
    x_min, y_min, x_max, y_max = map(float, b)
    w = max(0.0, x_max - x_min)
    h = max(0.0, y_max - y_min)
    return [float(x_min), float(y_min), float(w), float(h)]

def iou_xywh(a, b):
    ax1, ay1 = a[0], a[1]; ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]; bx2, by2 = b[0] + b[2], b[1] + b[3]
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def prepare_predictions(preds_raw, bbox_format):
    preds = []
    for p in preds_raw:
        if "bbox" not in p:
            raise ValueError("Każda predykcja musi mieć pole 'bbox'.")
        bbox = p["bbox"]
        if bbox_format == "xywh":
            bbox_coco = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        elif bbox_format == "xyxy":
            bbox_coco = xyxy_to_xywh(bbox)
        else:
            raise ValueError("INPUT_BBOX_FORMAT musi być 'xywh' lub 'xyxy'.")
        preds.append({
            "image_id": int(p["image_id"]),
            "category_id": int(p["category_id"]),
            "bbox": bbox_coco,
            "score": float(p.get("score", 1.0))
        })
    return preds

def basic_checks(gt_json, preds):
    if not all(k in gt_json for k in ("images","annotations","categories")):
        raise ValueError("GT JSON nie wygląda jak COCO (brakuje kluczy images/annotations/categories).")
    gt_image_ids = {img["id"] for img in gt_json["images"]}
    pred_image_ids = {p["image_id"] for p in preds}
    missing = pred_image_ids - gt_image_ids
    if missing:
        print("UWAGA: następujące image_id występują w predykcjach ale nie ma ich w GT (będą ignorowane):", list(missing)[:20])
    else:
        print("OK: Wszystkie image_id w predykcjach występują w GT.")
    gt_cat_ids = {c["id"] for c in gt_json["categories"]}
    pred_cat_ids = {p["category_id"] for p in preds}
    missing_cat = pred_cat_ids - gt_cat_ids
    if missing_cat:
        print("UWAGA: następujące category_id w predykcjach nie występują w GT:", missing_cat)
    else:
        print("OK: category_id zgodne z GT (przynajmniej nie ma braków).")

def run_coco_eval(gt_json_path, preds_json_path, max_dets=None):
    cocoGt = COCO(gt_json_path)
    cocoDt = cocoGt.loadRes(preds_json_path)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    if max_dets is not None:
        cocoEval.params.maxDets = max_dets
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval, cocoGt, cocoDt

def per_class_ap_from_cocoeval(cocoEval, cocoGt, iou_thr=None):
    precision = cocoEval.eval.get('precision', None)
    if precision is None:
        raise ValueError("Brak danych precision w cocoEval (może brak detekcji).")
    cat_ids = cocoEval.params.catIds
    cats = cocoGt.loadCats(cat_ids)
    id2name = {c["id"]: c["name"] for c in cats}
    ious = cocoEval.params.iouThrs
    if iou_thr is None:
        ap_per_cat = np.mean(precision[:, :, :, 0, -1], axis=(0,1))
    else:
        tidx = np.argmin(np.abs(ious - iou_thr))
        ap_per_cat = np.mean(precision[tidx, :, :, 0, -1], axis=0)
    rows = []
    for k, cid in enumerate(cat_ids):
        ap = ap_per_cat[k]
        rows.append({"category_id": int(cid), "category_name": id2name.get(cid, str(cid)), "AP": float(ap) if ap != -1 else float("nan")})
    return pd.DataFrame(rows)

def micro_match_and_pr(gt_json, preds, iou_for_match=0.5):
    gts_by_image = defaultdict(list)
    for ann in gt_json["annotations"]:
        gts_by_image[ann["image_id"]].append({
            "bbox": [float(x) for x in ann["bbox"]],
            "category_id": int(ann["category_id"]),
            "matched": False
        })
    total_gts = len(gt_json["annotations"])

    preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)
    tps = []
    fps = []
    matched_ious = []
    per_class_records = defaultdict(list)
    scores_list = []

    for p in preds_sorted:
        img_id = p["image_id"]
        pbox = p["bbox"]
        pcat = p["category_id"]
        pscore = p["score"]
        scores_list.append(pscore)

        gts = gts_by_image.get(img_id, [])
        best_iou = 0.0
        best_gt = None
        for gt in gts:
            if gt["category_id"] != pcat:
                continue
            if gt["matched"]:
                continue
            iou_v = iou_xywh(pbox, gt["bbox"])
            if iou_v > best_iou:
                best_iou = iou_v
                best_gt = gt
        if best_iou >= iou_for_match and best_gt is not None:
            tps.append(1); fps.append(0)
            matched_ious.append(best_iou)
            best_gt["matched"] = True
            per_class_records[pcat].append((pscore, 1, best_iou))
        else:
            tps.append(0); fps.append(1)
            per_class_records[pcat].append((pscore, 0, None))

    tps_cum = np.cumsum(tps)
    fps_cum = np.cumsum(fps)
    precisions = tps_cum / np.maximum(1, (tps_cum + fps_cum))
    recalls = (tps_cum / total_gts) if total_gts > 0 else np.zeros_like(tps_cum)

    if len(recalls) == 0:
        micro_ap = 0.0
    else:
        r = np.concatenate(([0.0], recalls))
        p = np.concatenate(([1.0], precisions))
        sidx = np.argsort(r)
        r_s = r[sidx]; p_s = p[sidx]
        micro_ap = float(np.trapezoid(p_s, r_s))

    per_class_pr = {}
    for cid, recs in per_class_records.items():
        if len(recs) == 0:
            per_class_pr[cid] = {"scores": [], "precision": [], "recall": [], "AP_micro": float("nan")}
            continue
        recs_sorted = sorted(recs, key=lambda x: x[0], reverse=True)
        scores = [r[0] for r in recs_sorted]
        tps_loc = np.array([r[1] for r in recs_sorted])
        fps_loc = 1 - tps_loc
        tps_c = np.cumsum(tps_loc); fps_c = np.cumsum(fps_loc)
        prec = tps_c / np.maximum(1, (tps_c + fps_c))
        total_gts_class = sum(1 for ann in gt_json["annotations"] if ann["category_id"] == cid)
        rec = tps_c / total_gts_class if total_gts_class > 0 else np.zeros_like(tps_c)
        if len(rec) == 0:
            apc = float("nan")
        else:
            r_c = np.concatenate(([0.0], rec))
            p_c = np.concatenate(([1.0], prec))
            sidx_c = np.argsort(r_c)
            apc = float(np.trapezoid(p_c[sidx_c], r_c[sidx_c]))
        per_class_pr[cid] = {"scores": scores, "precision": prec.tolist(), "recall": rec.tolist(), "AP_micro": apc}

    iou_stats = {}
    if len(matched_ious) > 0:
        iou_stats = {"count_matched": len(matched_ious), "mean_iou": float(np.mean(matched_ious)),
                     "median_iou": float(np.median(matched_ious)), "std_iou": float(np.std(matched_ious))}
    else:
        iou_stats = {"count_matched": 0, "mean_iou": None, "median_iou": None, "std_iou": None}

    return {
        "scores": scores_list,
        "precisions": precisions.tolist(),
        "recalls": recalls.tolist(),
        "micro_ap": micro_ap,
        "matched_ious": matched_ious,
        "iou_stats": iou_stats,
        "per_class_pr": per_class_pr,
        "total_gts": total_gts,
        "tps_cum": tps_cum.tolist(),
        "fps_cum": fps_cum.tolist(),
        "pred_count": len(preds_sorted)
    }

def compute_point_metrics(preds, gt_json, score_thr=0.5, iou_thr=0.5):
    preds_f = [p for p in preds if p["score"] >= score_thr]
    gts_by_image = defaultdict(list)
    for ann in gt_json["annotations"]:
        gts_by_image[ann["image_id"]].append({"bbox": ann["bbox"], "category_id": ann["category_id"], "matched": False})
    TP=0; FP=0
    for p in preds_f:
        img_gts = gts_by_image.get(p["image_id"], [])
        best_iou=0.0; best_gt=None
        for gt in img_gts:
            if gt["category_id"] != p["category_id"]:
                continue
            if gt["matched"]:
                continue
            iouv = iou_xywh(p["bbox"], gt["bbox"])
            if iouv > best_iou:
                best_iou = iouv; best_gt=gt
        if best_iou >= iou_thr and best_gt is not None:
            TP += 1; best_gt["matched"] = True
        else:
            FP += 1
    FN = sum(1 for img in gts_by_image.values() for gt in img if not gt.get("matched", False))
    precision = TP/(TP+FP) if (TP+FP)>0 else 0.0
    recall = TP/(TP+FN) if (TP+FN)>0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    return {"TP":TP, "FP":FP, "FN":FN, "precision":precision, "recall":recall, "f1":f1}

def plot_pr(recall, precision, out_path, title="PR Curve", show_ap=None):
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, marker='.', linewidth=1)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.grid(True)
    if show_ap is not None:
        plt.title(f"{title}  AP≈{show_ap:.4f}")
    else:
        plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_iou_hist(matched_ious, out_path, title="IoU histogram"):
    plt.figure(figsize=(6,4))
    if len(matched_ious) > 0:
        plt.hist(matched_ious, bins=30)
        plt.xlabel("IoU"); plt.ylabel("Count")
    else:
        plt.text(0.5, 0.5, "No matched detections", ha='center')
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def draw_boxes_on_image(image_path, gt_boxes=[], pred_boxes=[], out_path=None, max_size=1600):
    try:
        im = Image.open(image_path).convert("RGB")
    except Exception as e:
        return False
    w,h = im.size
    if max(w,h) > max_size:
        scale = max_size / max(w,h)
        new_w, new_h = int(w*scale), int(h*scale)
        im = im.resize((new_w,new_h))
        def scale_box(b): return [b[0]*scale, b[1]*scale, b[2]*scale, b[3]*scale]
        gt_boxes = [scale_box(b) for b in gt_boxes]
        pred_boxes = [scale_box(b) for b in pred_boxes]
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()
    # GT in green
    for box in gt_boxes:
        x,y,wbox,hbox = box
        draw.rectangle([x,y,x+wbox,y+hbox], outline="green", width=2)
    # Pred in red
    for box in pred_boxes:
        x,y,wbox,hbox = box
        draw.rectangle([x,y,x+wbox,y+hbox], outline="red", width=2)
    if out_path:
        im.save(out_path)
    return True



def main():
    out_dir = ensure_dir(OUTPUT_DIR)
    print("OUTPUT_DIR =", out_dir)
    gt = load_json(GT_JSON_PATH)
    preds_raw = load_json(PRED_JSON_PATH)
    preds = prepare_predictions(preds_raw, INPUT_BBOX_FORMAT)
    basic_checks(gt, preds)

    converted_preds_path = os.path.join(out_dir, "predictions_for_eval.json")
    save_json(preds, converted_preds_path)
    print("Zapisano przekonwertowane predykcje:", converted_preds_path)

    print("\nUruchamiam COCOeval...")
    cocoEval, cocoGt, cocoDt = run_coco_eval(GT_JSON_PATH, converted_preds_path, max_dets=MAX_DETS)
    stats_names = [
        "AP@[.5:.95]", "AP@0.5", "AP@0.75",
        "AP_small", "AP_medium", "AP_large",
        "AR@1", "AR@10", "AR@100",
        "AR_small", "AR_medium", "AR_large"
    ]
    coco_stats = {name: float(v) for name,v in zip(stats_names, cocoEval.stats.tolist())}

    df_map = per_class_ap_from_cocoeval(cocoEval, cocoGt, iou_thr=None)
    df_map.to_csv(os.path.join(out_dir, "per_class_ap_coco_map.csv"), index=False)
    df_ap50 = per_class_ap_from_cocoeval(cocoEval, cocoGt, iou_thr=0.5)
    df_ap50.to_csv(os.path.join(out_dir, "per_class_ap50.csv"), index=False)
    print("Zapisano per-class AP CSV.")

    print("\nLiczenie micro-eval (IoU = {}) ...".format(IOU_FOR_MICRO_EVAL))
    micro = micro_match_and_pr(gt, preds, iou_for_match=IOU_FOR_MICRO_EVAL)
    save_json(micro, os.path.join(out_dir, "micro_eval_full.json"))
    save_json({"matched_ious": micro.get("matched_ious", [])}, os.path.join(out_dir, "matched_ious_list.json"))

    df_overall_pr = pd.DataFrame({"recall": micro["recalls"], "precision": micro["precisions"], "rank": list(range(len(micro["recalls"])))})
    df_overall_pr.to_csv(os.path.join(out_dir, "overall_pr_points.csv"), index=False)

    per_class_points_dir = os.path.join(out_dir, "per_class_pr_csv")
    ensure_dir(per_class_points_dir)
    for cid, rec in micro["per_class_pr"].items():
        dfc = pd.DataFrame({"score": rec["scores"], "precision": rec["precision"], "recall": rec["recall"]})
        dfc.to_csv(os.path.join(per_class_points_dir, f"pr_class_{cid}.csv"), index=False)

    point_metrics = compute_point_metrics(preds, gt, score_thr=SCORE_THRESHOLD_FOR_POINT_METRICS, iou_thr=IOU_FOR_MICRO_EVAL)
    print("\nSingle-point metrics (score_thr={}):".format(SCORE_THRESHOLD_FOR_POINT_METRICS))
    print(point_metrics)

    fps_info = {}
    if PRED_TIMES_JSON:
        times = load_json(PRED_TIMES_JSON)
        if isinstance(times, dict):
            times_list = list(times.values()); n_images = len(times)
            total_time = sum([float(x) for x in times_list])
        else:
            total_time = sum([float(x.get("time_seconds",0.0)) for x in times])
            n_images = len(times)
        fps_info = {"total_inference_time_s": total_time, "n_images": n_images, "fps": (n_images/total_time) if total_time>0 else None,
                    "avg_latency_ms": (total_time/n_images)*1000.0 if n_images>0 else None}
    elif TOTAL_INFERENCE_TIME_SECONDS is not None:
        n_images = len(gt["images"])
        total_time = float(TOTAL_INFERENCE_TIME_SECONDS)
        fps_info = {"total_inference_time_s": total_time, "n_images": n_images, "fps": (n_images/total_time) if total_time>0 else None,
                    "avg_latency_ms": (total_time/n_images)*1000.0 if n_images>0 else None}
    else:
        fps_info["note"] = "Brak danych o czasie inferencji. Ustaw TOTAL_INFERENCE_TIME_SECONDS lub PRED_TIMES_JSON."

    precisions = np.array(micro["precisions"]); recalls = np.array(micro["recalls"])
    if len(precisions)>0 and len(recalls)>0:
        f1s = 2*precisions*recalls/np.maximum(1e-12,(precisions+recalls))
        best_idx = int(np.nanargmax(f1s))
        best_f1 = float(f1s[best_idx]); best_prec = float(precisions[best_idx]); best_rec = float(recalls[best_idx])
    else:
        best_f1 = best_prec = best_rec = None

    pr_out = os.path.join(out_dir, "overall_pr_curve.png")
    plot_pr(micro["recalls"], micro["precisions"], pr_out, title=f"Overall PR (IoU={IOU_FOR_MICRO_EVAL})", show_ap=micro["micro_ap"])
    plot_iou_hist(micro.get("matched_ious", []), os.path.join(out_dir, "iou_histogram.png"), title=f"IoU histogram matches@{IOU_FOR_MICRO_EVAL}")
    print("Zapisano wykresy PR i histogram IoU.")

    per_class_dir = os.path.join(out_dir, "per_class_pr")
    ensure_dir(per_class_dir)
    cats = {c["id"]: c["name"] for c in gt["categories"]}
    for cid, rec in micro["per_class_pr"].items():
        cname = cats.get(cid, str(cid))
        outp = os.path.join(per_class_dir, f"pr_class_{cid}_{cname}.png")
        if len(rec["precision"]) == 0:
            plt.figure(figsize=(4,3)); plt.text(0.5, 0.5, "No predictions for class", ha='center'); plt.title(f"{cname} (id={cid})")
            plt.savefig(outp, dpi=150); plt.close()
        else:
            plot_pr(rec["recall"], rec["precision"], outp, title=f"PR {cname}", show_ap=rec["AP_micro"])

    if EXPORT_FP_FN_IMAGES and IMAGES_DIR:
        print("Eksportuję top-N FP i FN obrazy (jeśli znalezione) ...")
        gt_by_image = defaultdict(list)
        for ann in gt["annotations"]:
            gt_by_image[ann["image_id"]].append(ann)
        preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)
        fp_records = []
        fn_records = []
        gt_matched_flags = defaultdict(list)
        gts_copy = defaultdict(list)
        for ann in gt["annotations"]:
            gts_copy[ann["image_id"]].append({"bbox": ann["bbox"], "category_id": ann["category_id"], "matched": False})

        for p in preds_sorted:
            imgid = p["image_id"]; pbox = p["bbox"]; pcat = p["category_id"]; pscore = p["score"]
            gts = gts_copy.get(imgid, [])
            best_iou=0.0; best_gt=None
            for gtitem in gts:
                if gtitem["category_id"] != pcat: continue
                if gtitem["matched"]: continue
                iov = iou_xywh(pbox, gtitem["bbox"])
                if iov > best_iou:
                    best_iou = iov; best_gt = gtitem
            if best_iou >= IOU_FOR_MICRO_EVAL and best_gt is not None:
                best_gt["matched"] = True
            else:
                fp_records.append({"pred": p, "iou": best_iou})
        for imgid, gts in gts_copy.items():
            for gtitem in gts:
                if not gtitem["matched"]:
                    fn_records.append({"image_id": imgid, "gt": gtitem})

        fp_out_dir = os.path.join(out_dir, "top_fp_images"); ensure_dir(fp_out_dir)
        fn_out_dir = os.path.join(out_dir, "top_fn_images"); ensure_dir(fn_out_dir)
        for i, rec in enumerate(fp_records[:EXPORT_TOP_N_FP]):
            p = rec["pred"]; imgid = p["image_id"]
            img_meta = next((im for im in gt["images"] if im["id"]==imgid), None)
            if img_meta is None: continue
            filename = img_meta.get("file_name")
            img_path = os.path.join(IMAGES_DIR, filename) if filename else None
            if img_path and os.path.exists(img_path):
                gts_here = [ann["bbox"] for ann in gt_by_image.get(imgid,[])]
                pred_box = p["bbox"]
                outp = os.path.join(fp_out_dir, f"FP_{i+1}_img{imgid}_score{p['score']:.3f}.jpg")
                draw_boxes_on_image(img_path, gt_boxes=gts_here, pred_boxes=[pred_box], out_path=outp)
        for i, rec in enumerate(fn_records[:EXPORT_TOP_N_FN]):
            imgid = rec["image_id"]
            img_meta = next((im for im in gt["images"] if im["id"]==imgid), None)
            if img_meta is None: continue
            filename = img_meta.get("file_name")
            img_path = os.path.join(IMAGES_DIR, filename) if filename else None
            if img_path and os.path.exists(img_path):
                gt_box = rec["gt"]["bbox"]
                outp = os.path.join(fn_out_dir, f"FN_{i+1}_img{imgid}.jpg")
                draw_boxes_on_image(img_path, gt_boxes=[gt_box], pred_boxes=[], out_path=outp)
        print("Eksport FP do:", fp_out_dir, "  FN do:", fn_out_dir)

    results_summary = {
        "coco_stats": coco_stats,
        "micro_ap": micro["micro_ap"],
        "micro_iou_stats": micro["iou_stats"],
        "single_point_metrics": point_metrics,
        "fps_info": fps_info,
        "best_f1_from_pr_curve": {"best_f1": best_f1, "precision": best_prec, "recall": best_rec},
        "config": {"IOU_FOR_MICRO_EVAL": IOU_FOR_MICRO_EVAL, "SCORE_THRESHOLD_FOR_POINT_METRICS": SCORE_THRESHOLD_FOR_POINT_METRICS}
    }
    save_json(results_summary, os.path.join(out_dir, "results_summary.json"))
    print("\nZapisano results_summary.json oraz pozostałe pliki w:", out_dir)

    print("\n--- Najważniejsze metryki ---")
    print("COCO mAP@[.5:.95]:", coco_stats.get("AP@[.5:.95]"))
    print("COCO AP@0.5:", coco_stats.get("AP@0.5"))
    print("Micro AP@IoU={}: {:.4f}".format(IOU_FOR_MICRO_EVAL, micro["micro_ap"]))
    print("Single-point Precision/Recall/F1 (score_thr={}):".format(SCORE_THRESHOLD_FOR_POINT_METRICS))
    print(point_metrics)
    print("FPS info:", fps_info)
    print("Best F1 on overall PR curve:", results_summary["best_f1_from_pr_curve"])
    print("\nGotowe.")

if __name__ == "__main__":
    main()






