
"""
Program generuje pliki z predykcjami bounding boxów w formacie YOLO11
(współrzędne są znormalizowane (0..1) względem rozmiaru obrazu)

Program tylko generuje pliki predykcji (bez porównywania z GT).
"""
from pathlib import Path
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------- CONFIG ---------------------------
MODEL_PATH = "modele_yolo/best.pt"   # <- ścieżka do modelu
IMAGES_DIR = "Dataset/Road_Objects_Dataset/YOLO11_format/test/images"
OUTPUT_DIR = "yolo_predictions"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]  # <- rozszerzenia obrazów do przetworzenia
CONF_THRESHOLD = 0.001
IOU_NMS = 0.45
DEVICE = "cpu"
RECURSIVE = False
SAVE_EMPTY_FILES = True
VERBOSE = True





def xyxy_to_xywhn(x1, y1, x2, y2, img_w, img_h):
    """Zamiana z absolute xyxy -> normalized xywh (center)"""
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return (cx / img_w, cy / img_h, w / img_w, h / img_h)


def list_images(directory: Path, exts=None, recursive=False):
    if exts is None:
        exts = [".jpg", ".jpeg", ".png"]
    if recursive:
        return sorted([p for p in directory.rglob("*") if p.suffix.lower() in exts and p.is_file()])
    else:
        return sorted([p for p in directory.iterdir() if p.suffix.lower() in exts and p.is_file()])



def main():
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("Błąd: wymagany pakiet 'ultralytics' nie jest zainstalowany lub nie można go załadować.")
        print("Zainstaluj: pip install ultralytics")
        raise e

    model_path = Path(MODEL_PATH)
    images_dir = Path(IMAGES_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Model nie istnieje: {model_path}")
        sys.exit(1)
    if not images_dir.exists():
        print(f"Katalog z obrazami nie istnieje: {images_dir}")
        sys.exit(1)

    if VERBOSE:
        print(f"Ładowanie modelu: {model_path}  (device={DEVICE})")
    model = YOLO(str(model_path))

    imgs = list_images(images_dir, exts=[e.lower() for e in IMAGE_EXTENSIONS], recursive=RECURSIVE)
    if len(imgs) == 0:
        print("Brak obrazów do przetworzenia.")
        return

    if VERBOSE:
        print(f"Znaleziono {len(imgs)} obraz(ów). Predykcje zostaną zapisane w: {output_dir}")

    for img_path in tqdm(imgs, desc="Inferencja", ncols=80):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                if VERBOSE:
                    print(f"Nie można wczytać obrazu: {img_path}")
                continue
            h, w = img.shape[:2]

            results = model.predict(source=img, conf=CONF_THRESHOLD, iou=IOU_NMS, device=DEVICE, verbose=False)
            if results is None or len(results) == 0:
                detections = []
            else:
                r = results[0]
                boxes = getattr(r, "boxes", None)
                detections = []
                if boxes is not None:
                    try:
                        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.numpy()
                        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf.numpy()
                        clses = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls.numpy()
                        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clses):
                            detections.append({
                                "class": int(cls),
                                "conf": float(conf),
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                            })
                    except Exception:
                        for b in boxes:
                            try:
                                x1, y1, x2, y2 = map(float, b.xyxy)
                                conf = float(b.conf)
                                cls = int(b.cls)
                                detections.append({
                                    "class": int(cls),
                                    "conf": float(conf),
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2,
                                })
                            except Exception:
                                continue
                else:
                    detections = []

            
            basename = img_path.stem
            out_file = output_dir / f"{basename}.txt"
            lines = []
            for det in detections:
                cx, cy, wn, hn = xyxy_to_xywhn(det["x1"], det["y1"], det["x2"], det["y2"], w, h)
                lines.append(f"{int(det['class'])} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f} {det['conf']:.6f}\n")

            if len(lines) == 0 and not SAVE_EMPTY_FILES:
                if out_file.exists():
                    out_file.unlink()
            else:
                with open(out_file, "w") as f:
                    f.writelines(lines)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if VERBOSE:
        print("Zakończono generowanie predykcji.")
        print(f"Pliki zapisane w: {output_dir}")
        print("Format plików: class x_center y_center width height confidence (wartości znormalizowane 0..1)")

if __name__ == "__main__":
    main()
