import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm

from Swin_DETR import DETR
from coco_cs import CocoDataset

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
    model.to(device)


    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    optimizer = torch.optim.AdamW(
        lr=train_config['lr'],
        params=filter(lambda p: p.requires_grad, model.parameters()),
        weight_decay=1E-4
    )

    lr_scheduler = MultiStepLR(
        optimizer,
        milestones=train_config['lr_steps'],
        gamma=0.1
    )

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
    if train_config.get('resume', False):
        ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                steps = checkpoint.get('steps', 0)
                best_val_total = checkpoint.get('best_val_total', float('inf'))
                epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
                print(f"[RESUME] Wznowiono od epoki {start_epoch}, steps={steps}")
            else:
                model.load_state_dict(checkpoint)
                print("[RESUME] Załadowano tylko wagi modelu (stary format ckpt)")



    for epoch in range(start_epoch, num_epochs):
        
        # TRAIN
        model.train()
        detr_classification_losses = []
        detr_localization_losses = []

        for idx, (ims, targets, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]")):
            for target in targets:
                target['boxes'] = target['boxes'].float().to(device)
                target['labels'] = target['labels'].long().to(device)

            images = torch.stack([im.float().to(device) for im in ims], dim=0)
            out = model(images, targets)
            batch_losses = out['loss']

            cls_loss = sum(batch_losses['classification'])
            loc_loss = sum(batch_losses['bbox_regression'])
            loss = (cls_loss + loc_loss)

            detr_classification_losses.append(cls_loss.item())
            detr_localization_losses.append(loc_loss.item())

            loss = loss / acc_steps
            loss.backward()


            if (idx + 1) % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if steps % train_config['log_steps'] == 0:
                print(f"DETR Classification Loss : {np.mean(detr_classification_losses):.4f} | "
                      f"DETR Localization Loss : {np.mean(detr_localization_losses):.4f} | "
                      f"lr: {lr_scheduler.get_last_lr()}")
                # TensorBoard log
                writer.add_scalar('train/cls_loss', np.mean(detr_classification_losses), steps)
                writer.add_scalar('train/loc_loss', np.mean(detr_localization_losses), steps)
                writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], steps)

            if torch.isnan(loss):
                print('Loss is becoming nan. Exiting')
                exit(0)

            steps += 1

        optimizer.step()
        optimizer.zero_grad()

        
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

        # Podsumowania epoki
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
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'steps': steps,
            'best_val_total': best_val_total,
            'epochs_no_improve': epochs_no_improve
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
    parser.add_argument('--config', dest='config_path', default='cs_coco.yaml', type=str)
    args = parser.parse_args()
    train(args)
