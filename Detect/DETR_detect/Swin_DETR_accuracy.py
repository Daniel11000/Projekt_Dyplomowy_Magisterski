
"""
Program porównuje pliki z predykcjami wytrenowanego modelu z Ground Truth (z datasetu)
oraz oblicza różne definicje "accuracy" i powiązane metryki.
"""
import os
import json
from collections import defaultdict
import math
import statistics

# ---------------------------- CONFIG ---------------------------
GT_JSON_PATH = r"Dataset/Road_Objects_Dataset/COCO_format/test/new_annotations.coco.json" # <-- ścieżka do ground-truth z Datasetu
PRED_JSON_PATH = r"Predictions/Se_DETR_pred/predictions_500_ep.json" # <-- ścieżka do predykcji modelu
OUTPUT_DIR = r"./accuracy_output" # <-- katalog wyjściowy

IOU_FOR_MATCH = 0.5
SCORE_THRESHOLD = 0.5
DEFAULT_SCORE_IF_MISSING = 0.0
APPLY_NMS = False
NMS_IOU = 0.5
BBOX_FORMAT = "xywh"
# ----------------------------------------------------------------



def load_json(path):
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

def prepare_predictions(preds_raw, bbox_format="xywh", default_score=0.0):
    preds = []
    missing_score_count = 0
    for p in preds_raw:
        if "bbox" not in p:
            raise ValueError("Każda predykcja musi mieć pole 'bbox'.")
        bbox = p["bbox"]
        if bbox_format == "xywh":
            bbox_coco = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        elif bbox_format == "xyxy":
            bbox_coco = xyxy_to_xywh(bbox)
        else:
            raise ValueError("BBOX_FORMAT musi być 'xywh' lub 'xyxy'.")
        score = p.get("score", None)
        if score is None:
            score = float(default_score)
            missing_score_count += 1
        preds.append({
            "image_id": int(p["image_id"]),
            "category_id": int(p["category_id"]),
            "bbox": bbox_coco,
            "score": float(score)
        })
    return preds, missing_score_count

def simple_nms(boxes_scores, iou_thr=0.5):
    if not boxes_scores:
        return []
    boxes_scores = sorted(boxes_scores, key=lambda x: x["score"], reverse=True)
    keep = []
    while boxes_scores:
        current = boxes_scores.pop(0)
        keep.append(current)
        remaining = []
        for b in boxes_scores:
            iou_v = iou_xywh(current["bbox"], b["bbox"])
            if iou_v <= iou_thr:
                remaining.append(b)
        boxes_scores = remaining
    return keep


def apply_optional_nms(preds, iou_thr=0.5):
    grouped = defaultdict(list)
    for p in preds:
        key = (p["image_id"], p["category_id"])
        grouped[key].append({"bbox": p["bbox"], "score": p["score"], "orig": p})
    kept = []
    for key, items in grouped.items():
        kept_items = simple_nms(items, iou_thr=iou_thr)
        for ki in kept_items:
            kept.append({
                "image_id": key[0],
                "category_id": key[1],
                "bbox": ki["bbox"],
                "score": ki["score"]
            })
    return kept

def greedy_match_counts(gt_annotations, preds, iou_thr=0.5, score_thr=0.5):
    gts_by_image = defaultdict(list)
    for i, ann in enumerate(gt_annotations):
        gts_by_image[ann["image_id"]].append({"idx": i, "bbox": [float(x) for x in ann["bbox"]], "category_id": int(ann["category_id"]), "matched": False})

    preds_filtered = [p for p in preds if p["score"] >= score_thr]
    preds_sorted = sorted(preds_filtered, key=lambda x: x["score"], reverse=True)

    matched_ious = []
    TP = 0; FP = 0

    for p in preds_sorted:
        img_id = p["image_id"]
        pcat = p["category_id"]
        pbox = p["bbox"]
        best_iou = 0.0
        best_gt = None
        gts = gts_by_image.get(img_id, [])
        for gt in gts:
            if gt["category_id"] != pcat:
                continue
            if gt["matched"]:
                continue
            iou_v = iou_xywh(pbox, gt["bbox"])
            if iou_v > best_iou:
                best_iou = iou_v
                best_gt = gt
        if best_gt is not None and best_iou >= iou_thr:
            TP += 1
            best_gt["matched"] = True
            matched_ious.append(best_iou)
        else:
            FP += 1

    FN = sum(1 for gts in gts_by_image.values() for gt in gts if not gt["matched"])
    return {"TP": TP, "FP": FP, "FN": FN, "matched_ious": matched_ious, "preds_considered": len(preds_filtered)}

def compute_image_level_exact_match(gt_annotations, preds, iou_thr=0.5, score_thr=0.5):
    gts_by_image = defaultdict(list)
    for ann in gt_annotations:
        gts_by_image[ann["image_id"]].append({"bbox": ann["bbox"], "category_id": ann["category_id"], "matched": False})
    preds_filtered = [p for p in preds if p["score"] >= score_thr]
    preds_by_image = defaultdict(list)
    for p in preds_filtered:
        preds_by_image[p["image_id"]].append(p)

    image_ids = sorted(list(set(list(gts_by_image.keys()) + list(preds_by_image.keys()))))
    exact_match_count = 0
    total_images = len(image_ids)

    for img_id in image_ids:
        gts = gts_by_image.get(img_id, [])
        preds_list = sorted(preds_by_image.get(img_id, []), key=lambda x: x["score"], reverse=True)
        for gt in gts:
            gt["matched"] = False
        fp_here = 0
        for p in preds_list:
            best_iou = 0.0; best_gt = None
            for gt in gts:
                if gt["category_id"] != p["category_id"]:
                    continue
                if gt["matched"]:
                    continue
                iov = iou_xywh(p["bbox"], gt["bbox"])
                if iov > best_iou:
                    best_iou = iov; best_gt = gt
            if best_gt is not None and best_iou >= iou_thr:
                best_gt["matched"] = True
            else:
                fp_here += 1
        fn_here = sum(1 for gt in gts if not gt["matched"])
        if (fp_here == 0) and (fn_here == 0):
            exact_match_count += 1

    exact_match_fraction = exact_match_count / total_images if total_images>0 else float("nan")
    return {"exact_match_count": exact_match_count, "total_images": total_images, "exact_match_fraction": exact_match_fraction}

def compute_presence_based_image_class_metrics(gt_annotations, preds, all_cat_ids, image_ids, score_thr=0.5):
    gt_presence = defaultdict(lambda: defaultdict(int))
    for ann in gt_annotations:
        gt_presence[ann["image_id"]][ann["category_id"]] += 1

    pred_presence = defaultdict(lambda: defaultdict(int))
    for p in preds:
        if p["score"] < score_thr:
            continue
        pred_presence[p["image_id"]][p["category_id"]] += 1

    per_class_counts = {cid: {"TP":0,"FP":0,"FN":0,"TN":0} for cid in all_cat_ids}
    for img in image_ids:
        for cid in all_cat_ids:
            g = 1 if gt_presence.get(img,{}).get(cid,0) > 0 else 0
            pr = 1 if pred_presence.get(img,{}).get(cid,0) > 0 else 0
            if g==1 and pr==1:
                per_class_counts[cid]["TP"] += 1
            elif g==0 and pr==1:
                per_class_counts[cid]["FP"] += 1
            elif g==1 and pr==0:
                per_class_counts[cid]["FN"] += 1
            else:
                per_class_counts[cid]["TN"] += 1

    per_class_metrics = {}
    sum_TP = sum_FP = sum_FN = sum_TN = 0
    for cid, c in per_class_counts.items():
        TP = c["TP"]; FP = c["FP"]; FN = c["FN"]; TN = c["TN"]
        denom = TP+FP+FN+TN
        accuracy = (TP+TN)/denom if denom>0 else float("nan")
        tpr = TP/(TP+FN) if (TP+FN)>0 else float("nan")  # recall
        tnr = TN/(TN+FP) if (TN+FP)>0 else float("nan")
        balanced = ((tpr if not math.isnan(tpr) else 0.0) + (tnr if not math.isnan(tnr) else 0.0))/2.0
        per_class_metrics[cid] = {"TP":TP,"FP":FP,"FN":FN,"TN":TN,"accuracy":accuracy,"tpr":tpr,"tnr":tnr,"balanced_accuracy":balanced}
        sum_TP += TP; sum_FP += FP; sum_FN += FN; sum_TN += TN

    total = sum_TP + sum_FP + sum_FN + sum_TN
    micro_accuracy = (sum_TP + sum_TN)/total if total>0 else float("nan")
    valid_accs = [m["accuracy"] for m in per_class_metrics.values() if not math.isnan(m["accuracy"])]
    macro_accuracy = sum(valid_accs)/len(valid_accs) if len(valid_accs)>0 else float("nan")
    valid_bal = [m["balanced_accuracy"] for m in per_class_metrics.values() if not math.isnan(m["balanced_accuracy"])]
    macro_balanced = sum(valid_bal)/len(valid_bal) if len(valid_bal)>0 else float("nan")

    return {"per_class_counts": per_class_counts, "per_class_metrics": per_class_metrics,
            "micro_accuracy_presence": micro_accuracy, "macro_accuracy_presence": macro_accuracy,
            "macro_balanced_accuracy_presence": macro_balanced,
            "global_counts": {"TP":sum_TP,"FP":sum_FP,"FN":sum_FN,"TN":sum_TN}}



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Wczytuję pliki...")
    gt = load_json(GT_JSON_PATH)
    preds_raw = load_json(PRED_JSON_PATH)

    preds, missing_score_count = prepare_predictions(preds_raw, bbox_format=BBOX_FORMAT, default_score=DEFAULT_SCORE_IF_MISSING)
    if missing_score_count > 0:
        print(f"Uwaga: {missing_score_count} predykcji nie miało pola 'score'. Ustawiono domyślny score = {DEFAULT_SCORE_IF_MISSING}.")
        print("Jeśli nie podano prawdziwych confidences, metryki zależne od rankingu (PR/AP) będą mniej informatywne.")

    if APPLY_NMS:
        print(f"Zastosuję prosty NMS przed ewaluacją (NMS_IOU = {NMS_IOU}) ...")
        preds = apply_optional_nms(preds, iou_thr=NMS_IOU)
        print(f"Po NMS liczba predykcji: {len(preds)}")

    gt_image_ids = {img["id"] for img in gt.get("images", [])}
    pred_image_ids = {p["image_id"] for p in preds}
    missing_img_ids = pred_image_ids - gt_image_ids
    if missing_img_ids:
        print("Uwaga: następujące image_id występują w predykcjach ale brak ich w GT (będą ignorowane przy niektórych metrykach):")
        print(list(missing_img_ids)[:30])

    gt_cat_ids = {c["id"] for c in gt.get("categories", [])}
    pred_cat_ids = {p["category_id"] for p in preds}
    missing_cat_ids = pred_cat_ids - gt_cat_ids
    if missing_cat_ids:
        print("Uwaga: następujące category_id występują w predykcjach ale nie ma ich w GT:", missing_cat_ids)

    gm = greedy_match_counts(gt.get("annotations", []), preds, iou_thr=IOU_FOR_MATCH, score_thr=SCORE_THRESHOLD)
    TP = gm["TP"]; FP = gm["FP"]; FN = gm["FN"]
    preds_considered = gm["preds_considered"]
    matched_ious = gm["matched_ious"]

    precision = TP/(TP+FP) if (TP+FP)>0 else float("nan")
    recall = TP/(TP+FN) if (TP+FN)>0 else float("nan")
    f1 = 2*precision*recall/(precision+recall) if (not math.isnan(precision) and not math.isnan(recall) and (precision+recall)>0) else float("nan")
    detection_accuracy = TP / (TP + FP + FN) if (TP+FP+FN)>0 else float("nan")

    im_exact = compute_image_level_exact_match(gt.get("annotations", []), preds, iou_thr=IOU_FOR_MATCH, score_thr=SCORE_THRESHOLD)

    categories = gt.get("categories", [])
    all_cat_ids = sorted([c["id"] for c in categories])
    image_ids = sorted([img["id"] for img in gt.get("images", [])])
    presence = compute_presence_based_image_class_metrics(gt.get("annotations", []), preds, all_cat_ids, image_ids, score_thr=SCORE_THRESHOLD)

    if matched_ious:
        mean_iou = statistics.mean(matched_ious)
        median_iou = statistics.median(matched_ious)
        std_iou = statistics.pstdev(matched_ious) if len(matched_ious)>1 else 0.0
    else:
        mean_iou = median_iou = std_iou = None

    summary = {
        "config": {
            "GT_JSON_PATH": GT_JSON_PATH,
            "PRED_JSON_PATH": PRED_JSON_PATH,
            "IOU_FOR_MATCH": IOU_FOR_MATCH,
            "SCORE_THRESHOLD": SCORE_THRESHOLD,
            "DEFAULT_SCORE_IF_MISSING": DEFAULT_SCORE_IF_MISSING,
            "APPLY_NMS": APPLY_NMS,
            "NMS_IOU": NMS_IOU,
            "BBOX_FORMAT": BBOX_FORMAT
        },
        "object_level_counts": {"TP": TP, "FP": FP, "FN": FN, "preds_considered": preds_considered},
        "object_level_metrics": {"precision": precision, "recall": recall, "f1": f1, "detection_accuracy_TP_over_TP_FP_FN": detection_accuracy},
        "image_level_exact_match": im_exact,
        "presence_image_class_metrics_summary": {
            "micro_accuracy_presence": presence["micro_accuracy_presence"],
            "macro_accuracy_presence": presence["macro_accuracy_presence"],
            "macro_balanced_accuracy_presence": presence["macro_balanced_accuracy_presence"],
            "global_counts": presence["global_counts"]
        },
        "iou_stats_for_matched": {"count_matched": len(matched_ious), "mean_iou": mean_iou, "median_iou": median_iou, "std_iou": std_iou}
    }

    out_path = os.path.join(OUTPUT_DIR, "accuracy_summary.json")
    save_json(summary, out_path)
    print("\nZapisano summary do:", out_path)


    print("\n--- KRÓTKI RAPORT ---")
    print(f"TP / FP / FN = {TP} / {FP} / {FN}  (preds considered with score >= {SCORE_THRESHOLD}: {preds_considered})")
    print(f"Precision = TP/(TP+FP) = {precision:.4f}")
    print(f"Recall    = TP/(TP+FN) = {recall:.4f}")
    print(f"F1        = 2*P*R/(P+R) = {f1:.4f}")
    print(f"Detection accuracy (TP/(TP+FP+FN)) = {detection_accuracy:.4f}")
    print(f"IoU (matched): count={len(matched_ious)}, mean={mean_iou}, median={median_iou}, std={std_iou}")
    print(f"Image exact-match fraction = {im_exact['exact_match_fraction']:.4f} ({im_exact['exact_match_count']}/{im_exact['total_images']})")
    print(f"Presence-based micro accuracy (image x class) = {presence['micro_accuracy_presence']:.4f}")
    print(f"Presence-based macro accuracy (average per-class) = {presence['macro_accuracy_presence']:.4f}")
    print(f"Presence-based macro balanced accuracy = {presence['macro_balanced_accuracy_presence']:.4f}")

    print("\n--- OPIS METRYK (krótko) ---")
    print("1) Precision = TP / (TP + FP). Miara: z wszystkich detekcji, jaki odsetek to prawdziwe trafienia.")
    print("2) Recall = TP / (TP + FN). Miara: z wszystkich rzeczywistych obiektów, jaki odsetek model wykrył.")
    print("3) F1 = harmoniczna precision i recall. Przydatne do zrównoważenia obu miar.")
    print("4) Detection accuracy (TP/(TP+FP+FN)) = prosty jednowymiarowy wskaźnik: jaki ułamek wszystkich przypadków detekcyjnych to TP.")
    print("   Uwaga: to NIE jest klasyczna accuracy (brakuje TN). Nie zaleca się używania go jako jedynej metryki w detekcji.")
    print("5) Image-level exact match: odsetek obrazów, dla których model nie ma ani FP, ani FN (bardzo restrykcyjne).")
    print("6) Presence-based (image x class) accuracy: budujemy macierz konfuzji dla par (image, class).")
    print("   Dla każdej pary mamy TP/FP/FN/TN (czy klasa występuje w obrazie). Dzięki temu możemy policzyć klasyczną accuracy = (TP+TN)/(TP+FP+FN+TN).")
    print("   - micro accuracy: agregowane liczone po wszystkich parach (ważone przez liczbę przykładów).")
    print("   - macro accuracy: najpierw liczymy accuracy per class, potem uśredniamy (równoważone między klasami).")
    print("   - balanced accuracy: (TPR + TNR)/2 per class, potem średnia (lepsze przy odkształconych klasach).")

    print("\n--- UWAGI / Rekomendacje ---")
    print("- W detekcji preferowane metryki to AP (mAP), precision/recall i PR-curve; accuracy w sensie klasycznym jest rzadko używana.")
    print("- Jeśli predykcje nie mają sensownych 'score', ustawienie DEFAULT_SCORE_IF_MISSING może zafałszować wyniki. Zalecane: wygenerować rzeczywiste confidences z modelu.")
    print("- Jeśli predykcje zawierają duplikaty (brak NMS), rozważ ustawienie APPLY_NMS=True przed ewaluacją.")
    print("- Do publikacji dołącz: per-class AP (COCOeval), PR curve, przykłady TP/FP/FN i opis procedury matching (iou_thr, score_thr, NMS).")

if __name__ == "__main__":
    main()
