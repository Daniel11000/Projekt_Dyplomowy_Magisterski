import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm


from Swin_DETR import DETR
from coco_fine_tuning import CocoDataset

from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')

if device == torch.device('cuda'):
    print('Training on GPU')
if device == torch.device('cpu'):
    print('Training on CPU')


import torch.nn as nn
def load_pretrained_partial(model, ckpt_path, device='cpu'):
    """
    Wczytuje checkpoint i podstawia tylko parametry które pasują kształtem
    (bez wyrzucania błędów dla headów z inną liczbą klas).
    """
    ck = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in ck:
        sd = ck['model_state_dict']
    else:
        sd = ck
    model_sd = model.state_dict()
    loaded = {}
    skipped = []
    for k, v in sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            loaded[k] = v
        else:
            skipped.append(k)
    model_sd.update(loaded)
    model.load_state_dict(model_sd)
    print(f"[LOAD] Załadowano {len(loaded)} parametrów, pominieto {len(skipped)} parametrów.")
    if len(skipped) > 0:
        print("Przykładowe pominiete klucze:", skipped[:20])


def collate_function(data):
    return tuple(zip(*data))


def train(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    train_config = config['train_params']
    model_config = config['model_params']

    def _to_float_if_str(x):
        if isinstance(x, str):
            try:
                return float(x)
            except Exception:
                return x
        return x

    def _to_int_if_str(x):
        if isinstance(x, str):
            try:
                return int(x) if x.isdigit() else int(float(x))
            except Exception:
                return x
        return x

    for k in ['lr', 'backbone_lr', 'warmup_factor']:
        if k in train_config:
            train_config[k] = _to_float_if_str(train_config[k])

    for k in ['batch_size', 'num_epochs', 'acc_steps', 'save_every', 'warmup_epochs', 'log_steps', 'num_workers']:
        if k in train_config:
            train_config[k] = _to_int_if_str(train_config[k])

    if 'lr_steps' in train_config:
        train_config['lr_steps'] = [int(x) for x in train_config['lr_steps']]

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- DANE TRENINGOWE ---
    coco_train = CocoDataset(
        ann_file=dataset_config['train_ann_file'],
        img_dir=dataset_config['train_img_dir'],
        im_size=dataset_config['im_size'],
        split='train'
    )
    train_loader = DataLoader(
        coco_train,
        batch_size=train_config['batch_size'],
        shuffle=True,
        collate_fn=collate_function
    )

    # --- DANE WALIDACYJNE ---
    val_ann = dataset_config.get('val_ann_file', dataset_config['train_ann_file'])
    val_img = dataset_config.get('val_img_dir', dataset_config['train_img_dir'])
    val_split = dataset_config.get('val_split', 'val')  # jeśli dataset nie ma 'val', ustaw na 'train'
    coco_val = CocoDataset(
        ann_file=val_ann,
        img_dir=val_img,
        im_size=dataset_config['im_size'],
        split=val_split
    )
    val_loader = DataLoader(
        coco_val,
        batch_size=train_config.get('val_batch_size', train_config['batch_size']),
        shuffle=False,
        collate_fn=collate_function
    )

    # --- MODEL ---
    inferred_num_classes = coco_train.num_classes + 1  # +1 for BG (index 0)
    cfg_num_classes = dataset_config.get('num_classes', None)
    if cfg_num_classes is None:
        model_num_classes = inferred_num_classes
    else:
        if cfg_num_classes != inferred_num_classes:
            print(f"Warning: dataset reports {coco_train.num_classes} classes; "
                  f"cfg had num_classes={cfg_num_classes}. Using inferred value {inferred_num_classes}.")
        model_num_classes = inferred_num_classes

    model = DETR(
        config=model_config,
        num_classes=model_num_classes,
        bg_class_idx=dataset_config.get('bg_class_idx', 0)
    )


    model.class_mlp = nn.Linear(model.d_model, model_num_classes)
    nn.init.xavier_uniform_(model.class_mlp.weight)
    if model.class_mlp.bias is not None:
        nn.init.constant_(model.class_mlp.bias, 0.0)

    model.bbox_mlp = nn.Sequential(
        nn.Linear(model.d_model, model.d_model),
        nn.ReLU(),
        nn.Linear(model.d_model, model.d_model),
        nn.ReLU(),
        nn.Linear(model.d_model, 4),
    )
    for m in model.bbox_mlp.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    model.to(device)

    pretrained_ckpt = train_config.get('pretrained_ckpt', None)
    if pretrained_ckpt:
        if os.path.exists(pretrained_ckpt):
            load_pretrained_partial(model, pretrained_ckpt, device=device)
        else:
            print(f"[WARN] pretrained_ckpt {pretrained_ckpt} not found.")


    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    freeze_epochs = train_config.get('freeze_backbone_epochs', 10)

    optimizer = None
    lr_scheduler = None

    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    steps = 0

    early_stopping_patience = train_config.get('early_stopping_patience', 50)
    warmup_epochs = train_config.get('warmup_epochs', 350)
    warmup_factor = train_config.get('warmup_factor', 1e-8)

    save_every = train_config.get('save_every', 50)  # co ile epok zapisać osobny checkpoint
    start_epoch = 0
    new_best_ckpt_path = os.path.join(train_config['task_name'], f"best_{train_config['ckpt_name']}")



    best_val_total = float('inf')

    base_name, ext = os.path.splitext(train_config['ckpt_name'])
    best_ckpt_filename = f"best_{base_name}_weight{ext}"
    best_ckpt_path = os.path.join(train_config['task_name'], best_ckpt_filename)


    epochs_no_improve = 0
    epochs_no_improve_after_warmup = 0

    task_dir = train_config['task_name']
    os.makedirs(task_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(task_dir, 'TensorBoard_log'))


    # --- Resume ---
    start_epoch = 0
    steps = 0
    best_val_total = float('inf')
    epochs_no_improve = 0
    backbone_frozen_flag = None

    if train_config.get('resume', False):
        ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            # wczytywanie wag modelu
            if 'model_state_dict' in ckpt:
                try:
                    model.load_state_dict(ckpt['model_state_dict'])
                    print("[RESUME] model weights loaded from checkpoint.")
                except RuntimeError as e:
                    print(f"[RESUME][WARN] model.load_state_dict failed: {e}. Attempting strict=False load.")
                    model.load_state_dict(ckpt['model_state_dict'], strict=False)
                start_epoch = ckpt.get('epoch', 0) + 1
                steps = ckpt.get('steps', 0)
                best_val_total = ckpt.get('best_val_total', float('inf'))
                epochs_no_improve = ckpt.get('epochs_no_improve', 0)
                backbone_frozen_flag = ckpt.get('backbone_frozen', None)
                print(f"[RESUME] start_epoch={start_epoch}, steps={steps}, backbone_frozen_flag={backbone_frozen_flag}")
            else:
                try:
                    model.load_state_dict(ckpt)
                    print("[RESUME] załadowano starszy format wag.")
                except Exception as e:
                    print(f"[RESUME][WARN] Nie udało się załadować starszego formatu wag: {e}")
        else:
            print(f"[RESUME] resume True, but checkpoint {ckpt_path} not found. Starting from scratch.")
        



    if backbone_frozen_flag is not None:
        if backbone_frozen_flag:
            for p in model.backbone.parameters():
                p.requires_grad = False
        else:
            for p in model.backbone.parameters():
                p.requires_grad = True
    else:
        if start_epoch < freeze_epochs:
            for p in model.backbone.parameters():
                p.requires_grad = False
        else:
            for p in model.backbone.parameters():
                p.requires_grad = True

    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n]
    head_params = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': head_params, 'lr': train_config['lr']},
        {'params': backbone_params, 'lr': train_config.get('backbone_lr', 1e-5)}
    ], weight_decay=1E-4)

    lr_scheduler = MultiStepLR(
        optimizer,
        milestones=train_config.get('lr_steps', []),
        gamma=0.1
    )

    if train_config.get('resume', False) and os.path.exists(os.path.join(train_config['task_name'], train_config['ckpt_name'])):
        ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print("[RESUME] optimizer state loaded.")
            except Exception as e:
                print(f"[RESUME][WARN] Could not load optimizer state: {e}")
        if 'scheduler_state_dict' in ckpt and ckpt.get('scheduler_state_dict', None):
            try:
                lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                print("[RESUME] lr_scheduler state loaded.")
            except Exception as e:
                print(f"[RESUME][WARN] Could not load lr_scheduler state: {e}")




    for epoch in range(start_epoch, num_epochs):


        if epoch == 0:
            print(f"[INFO] Backbone frozen for first {freeze_epochs} epochs (freeze_backbone_epochs={freeze_epochs})")
        if epoch == freeze_epochs:
            print(f"[UNFREEZE] Odzamrazam backbone na epoch {epoch}. Tworze nowy optimizer z mniejszym LR dla backbone.")
            for p in model.backbone.parameters():
                p.requires_grad = True

            backbone_lr = train_config.get('backbone_lr', 1e-5)
            optimizer = torch.optim.AdamW([
                {'params': [p for n,p in model.named_parameters() if 'backbone' not in n and p.requires_grad], 'lr': train_config['lr']},
                {'params': [p for n,p in model.named_parameters() if 'backbone' in n], 'lr': backbone_lr}
            ], weight_decay=1E-4)


            try:
                lr_scheduler = MultiStepLR(optimizer, milestones=train_config.get('lr_steps', []), gamma=0.1)
            except Exception as e:
                print(f"[WARN] Nie udało się odtworzyć scheduler: {e}")


        # TRAIN
        model.train()
        detr_classification_losses = []
        detr_localization_losses = []

        accum_counter = 0
        max_grad_norm = train_config.get('grad_clip', None)

        for idx, (ims, targets, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]")):
            for target in targets:
                target['boxes'] = target['boxes'].float().to(device)
                target['labels'] = target['labels'].long().to(device)

            images = torch.stack([im.float().to(device) for im in ims], dim=0)
            out = model(images, targets)
            batch_losses = out['loss']

            cls_loss = sum(batch_losses['classification'])
            loc_loss = sum(batch_losses['bbox_regression'])
            loss = cls_loss + loc_loss

            detr_classification_losses.append(cls_loss.item())
            detr_localization_losses.append(loc_loss.item())

            loss = loss / acc_steps
            loss.backward()
            accum_counter += 1


            if (idx + 1) % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if torch.isnan(loss):
                print(f"[ERROR] Loss is NaN at epoch {epoch+1}, batch {idx}. Saving debug ckpt.")
                debug_ckpt = {
                    'epoch': epoch,
                    'batch_idx': idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                    'steps': steps
                }
                debug_path = os.path.join(train_config['task_name'], f"debug_nan_epoch{epoch+1}_batch{idx}.pth")
                try:
                    torch.save(debug_ckpt, debug_path)
                    print(f"[DEBUG] Saved debug checkpoint to {debug_path}")
                except Exception as e:
                    print(f"[WARN] Nie udało się zapisać debug ckpt: {e}")
                optimizer.zero_grad()
                break

            if accum_counter == acc_steps:
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                accum_counter = 0

            if steps % train_config['log_steps'] == 0:
                print(f"DETR Classification Loss : {np.mean(detr_classification_losses):.4f} | "
                      f"DETR Localization Loss : {np.mean(detr_localization_losses):.4f} | "
                      f"lr: {lr_scheduler.get_last_lr()}")
                writer.add_scalar('train/cls_loss', np.mean(detr_classification_losses), steps)
                writer.add_scalar('train/loc_loss', np.mean(detr_localization_losses), steps)
                writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], steps)

            if torch.isnan(loss):
                print('Loss is becoming nan. Exiting')
                exit(0)

            steps += 1


        if accum_counter > 0:
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            accum_counter = 0

        
        # VAL
        model.eval()
        val_cls_losses = []
        val_loc_losses = []
        with torch.no_grad():
            for ims, targets, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]"):
                for target in targets:
                    target['boxes'] = target['boxes'].float().to(device)
                    target['labels'] = target['labels'].long().to(device)

                images = torch.stack([im.float().to(device) for im in ims], dim=0)
                out = model(images, targets)
                batch_losses = out['loss']

                cls_loss = sum(batch_losses['classification'])
                loc_loss = sum(batch_losses['bbox_regression'])
                val_cls_losses.append(cls_loss.item())
                val_loc_losses.append(loc_loss.item())


        # Podsumowanie epoki
        train_cls_mean = float(np.mean(detr_classification_losses)) if detr_classification_losses else 0.0
        train_loc_mean = float(np.mean(detr_localization_losses)) if detr_localization_losses else 0.0
        val_cls_mean = float(np.mean(val_cls_losses)) if val_cls_losses else 0.0
        val_loc_mean = float(np.mean(val_loc_losses)) if val_loc_losses else 0.0

        print(f'Finished epoch {epoch+1}')
        print(f"TRAIN  | DETR Classification Loss : {train_cls_mean:.4f} | DETR Localization Loss : {train_loc_mean:.4f}")
        print(f"VAL    | DETR Classification Loss : {val_cls_mean:.4f} | DETR Localization Loss : {val_loc_mean:.4f}")

        # TensorBoard: log końca epoki
        writer.add_scalar('epoch/train_cls_loss', train_cls_mean, epoch+1)
        writer.add_scalar('epoch/train_loc_loss', train_loc_mean, epoch+1)
        writer.add_scalar('epoch/val_cls_loss', val_cls_mean, epoch+1)
        writer.add_scalar('epoch/val_loc_loss', val_loc_mean, epoch+1)


        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else {},
            'steps': steps,
            'best_val_total': best_val_total,
            'epochs_no_improve': epochs_no_improve,
            'backbone_frozen': any(not p.requires_grad for p in model.backbone.parameters())
        }
        torch.save(ckpt, os.path.join(train_config['task_name'], train_config['ckpt_name']))

        # --- Zapis co N epok ---
        if (epoch + 1) % save_every == 0:
            extra_ckpt_path = os.path.join(
                train_config['task_name'], f"epoch_{epoch+1}_{train_config['ckpt_name']}")
            torch.save(ckpt, extra_ckpt_path)
            print(f"[SAVE] Zapisano checkpoint: {extra_ckpt_path}")

        # --- Early stopping ---
        val_total = val_cls_mean + val_loc_mean
        if val_total < best_val_total:
            best_val_total = val_total
            torch.save(ckpt, new_best_ckpt_path)

            torch.save(model.state_dict(), best_ckpt_path)

            print(f"[BEST] Zapisano nowy najlepszy model (val_total={best_val_total:.4f})")
            epochs_no_improve = 0
            epochs_no_improve_after_warmup = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{early_stopping_patience} epochs")

            if epoch >= warmup_epochs:
                epochs_no_improve_after_warmup = 0
            else:
                epochs_no_improve_after_warmup += 1
                print(f"No improvement (after Warm-up) for {epochs_no_improve_after_warmup}/{early_stopping_patience} epochs")

            if epochs_no_improve >= early_stopping_patience:
                print(f"   Early stopping:")
                if epochs_no_improve >= early_stopping_patience and epoch >= warmup_epochs:
                    print(f"   Should be Stopped but patience counts after the warm-up epochs{epoch + 1}")
                    if epochs_no_improve_after_warmup >= early_stopping_patience and epoch >= warmup_epochs:
                        print(f"[STOP] Early stopping triggered at epoch {epoch+1}")
                        break
                else:
                    print(f"[not stopped] Early stopping was not triggered at epoch {epoch + 1} because of warmup")



        if epoch < warmup_epochs:
            warmup_lr = train_config['lr'] * ((epoch + 1) / warmup_epochs)
            for g in optimizer.param_groups:
                g['lr'] = warmup_lr
        else:
            if lr_scheduler is not None:
                lr_scheduler.step()



        base_name, ext = os.path.splitext(train_config['ckpt_name'])
        new_ckpt_name = base_name + "_weight" + ext
        last_ckpt_path = os.path.join(train_config['task_name'], new_ckpt_name)
        torch.save(model.state_dict(), last_ckpt_path)


        val_total = val_cls_mean + val_loc_mean
        if val_total < best_val_total:
            best_val_total = val_total
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"Saved BEST checkpoint at {best_ckpt_path} (val_total: {best_val_total:.4f})")

    
    writer.close()
    
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for detr training')
    parser.add_argument('--config', dest='config_path', default='coco_fine_tuning.yaml', type=str)
    args = parser.parse_args()
    train(args)
