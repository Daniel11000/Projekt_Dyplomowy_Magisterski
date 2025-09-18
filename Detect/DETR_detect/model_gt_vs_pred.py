# Program rysuje na jednym obrazku przewidywania modelu oraz Ground Truth z Datasetu

import os
import cv2
import json
from pycocotools.coco import COCO


# ---------------------------- CONFIG ---------------------------
GT_JSON = r"Dataset/Road_Objects_Dataset/COCO_format/test/new_annotations.coco.json" # <-- ścieżka do ground-truth z Datasetu
PRED_JSON = r"Predictions/Se_DETR_pred/predictions_500_ep.json"
IMAGES_DIR = r"Dataset/Road_Objects_Dataset/COCO_format/test"


gt    = COCO(GT_JSON)
preds = json.load(open(PRED_JSON))

img_ids  = gt.getImgIds()
num_imgs = len(img_ids)
idx      = 0

win_name = "GT vs Pred"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

def draw_annotations(im, image_id):
    anns = gt.loadAnns(gt.getAnnIds(imgIds=image_id))
    for a in anns:
        x,y,w,h = map(int, a['bbox'])
        cv2.rectangle(im, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.putText(im, f"GT:{a['category_id']}", (x,y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    for p in preds:
        if p['image_id'] == image_id:
            x,y,w,h = p['bbox']
            cv2.rectangle(im, (int(x),int(y)),
                          (int(x+w),int(y+h)), (0,255,0), 2)
            cv2.putText(im,
                        f"P:{p['category_id']} {p['score']:.2f}",
                        (int(x),int(y+h+12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

while True:
    img_info = gt.loadImgs(img_ids[idx])[0]
    path     = os.path.join(IMAGES_DIR, img_info['file_name'])
    im       = cv2.imread(path)

    if im is None:
        print(f"[ERROR] Nie udało się wczytać obrazu:\n  {path}")
        print("Sprawdź ścieżki i uprawnienia dostępu.")
        break
    else:
        h, w = im.shape[:2]
        print(f"[DEBUG] ({idx+1}/{num_imgs}) {img_info['file_name']} - {w}x{h}")

    draw_annotations(im, img_info['id'])

    label = f"{idx+1}/{num_imgs} : {img_info['file_name']}"
    cv2.putText(im, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow(win_name, im)
    key = cv2.waitKey(0)


    if key == 27 or key == ord('q'):      # ESC lub q
        break
    elif key == 2424832 or key == ord('a'):  # <- lub a
        idx = max(0, idx-1)
    elif key == 2555904 or key == ord('d'):  # -> lub d
        idx = min(num_imgs-1, idx+1)
    else:
        continue

cv2.destroyAllWindows()

