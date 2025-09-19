import os
import torch
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Urządzenie: {device}')
    return device

def train_yolo11n(
    data_yaml: str,
    project_dir: str,
    pretrained: str,
    epochs: int = 50,
    augment: bool = True,
    patience: int = 10,
    batch: int = 8,
    img_size: int = 640
):
    
    device = get_device()

    # 1) Załaduj model z wagami wstępnymi
    model = YOLO(pretrained)  # ładuje architekturę i wagi
    model.to(device)

    best_map = 0.0
    best_epoch = 0
    no_improve = 0

    # 2) Petla trening + walidacja
    for epoch in range(1, epochs + 1):
        print(f'\n=== Epoka {epoch}/{epochs} ===')

        # -- trening
        #    'exist_ok=True' pozwala na kontynuację w tym samym katalogu
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            augment=augment,
            batch=batch,
            project=project_dir,
            workers=0,
            name='run',
            exist_ok=True
        )

        # -- walidacja na zbiorze walidacyjnym
        results = model.val(
            data=data_yaml,
            imgsz=img_size,
            project=project_dir,
            batcg=batch,
            workers=0,
            name='run-val',
            exist_ok=True
        )

        # Wyciągnięcie głównych metryk z obiektu results
        # (zaklozono, ze results.metrics.map50_95 to mAP@0.5:0.95)
        current_map = results.metrics.map50_95
        box_loss    = results.metrics.box_loss
        cls_loss    = results.metrics.cls_loss
        obj_loss    = results.metrics.obj_loss
        lr          = results.params.lr

        # -- metryki
        print(f'lr={lr:.3e}  box_loss={box_loss:.3f}  cls_loss={cls_loss:.3f}  '
              f'obj_loss={obj_loss:.3f}  mAP@0.5:0.95={current_map:.3f}')

        # -- eczesne zatrzymanie (early stopping)
        if current_map > best_map:
            best_map   = current_map
            best_epoch = epoch
            no_improve = 0
            # zapis najlepszego modelu
            best_weights = os.path.join(project_dir, 'run', f'best_epoch{epoch}.pt')
            model.save(best_weights)
            print(f'[INFO] Nowy najlepszy model ({best_map:.3f}), zapisano: {best_weights}')
        else:
            no_improve += 1
            print(f'[INFO] Brak poprawy ({no_improve}/{patience})')

        if no_improve >= patience:
            print(f'\n[INFO] Przerwanie treningu po {patience} epokach bez poprawy.')
            break

    # 3) Podsumowanie
    print('\n=== Podsumowanie treningu ===')
    print(f'Najlepsze mAP@0.5:0.95 = {best_map:.3f} (epoka {best_epoch})')
    print(f'Wagi najlepszego modelu: {best_weights}')

    return best_weights

def main():
    data_yaml_path = 'Dataset/Road_Objects_Dataset/YOLO11_format/data.yaml'
    output_project = 'results'
    pretrained_weights = 'yolo11n.pt'
    # augment_flag = True
    augment_flag = False
    max_epochs = 10000
    patience = 50
    batch = 4

    # 2) Uruchomienie funkcji treningowej
    best_model_path=train_yolo11n(
        data_yaml=data_yaml_path,
        project_dir=output_project,
        pretrained=pretrained_weights,
        epochs=max_epochs,
        augment=augment_flag,
        patience=patience,
        batch=batch,
        img_size=640
    )

    print(f'\n[MAIN] Trening zakończony. Najlepszy model: {best_model_path}')

if __name__ == '__main__':
    main()
