# Program służy do sprawdzenia jak model sobie radzi na konkretnym obrazku.

import cv2
from ultralytics import YOLO

# Lista klas
classes = [
    'B1', 'B12', 'B13', 'B16', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'BC1', 'BC10', 'BC2', 'BC3',
    'BC7', 'BC8', 'Bb1', 'Bb2', 'Biker', 'Bus', 'Bz1', 'C1', 'C10', 'C2', 'C3', 'C4', 'C5', 'C6',
    'C7', 'C8', 'Car', 'Cb1', 'Cb2', 'CityDirSign', 'CitySign', 'Crossing', 'Cw1', 'Cz1', 'L',
    'Motorbike', 'Ostp', 'P1', 'P2', 'P3', 'Person', 'R', 'Rarrow', 'Rstop', 'S1', 'S2',
    'StreetSign', 'T1', 'T10', 'T11', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T2', 'T20', 'T3',
    'T4', 'T5', 'T6', 'T9', 'TL', 'TLb', 'TS', 'TSb', 'Tram', 'Truck', 'Y1', 'Y2', 'arrowL', 'dirL',
    'dirR', 'eosL', 'gL', 'i1', 'i2', 'i3', 'iBar', 'iPlate', 'iPlateB', 'iTable', 'rL', 'rcL', 'ryL', 'yL'
]


# Załaduj model
model_path = "Models/yolo_results/run/weights/best.pt"

model = YOLO(model_path)

# Wczytaj obraz
image_path = 'img/obrazek.jpg'

img = cv2.imread(image_path)


results = model(img, verbose=False)[0]

for i in range(len(results.boxes)):
    box = results.boxes[i]

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    label = f"{classes[cls_id]} ({conf:.2f})"

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
    cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Pokaż wynik
cv2.imshow("Detected Objects", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

