import os
import sys
import copy

# Inject root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import time
import argparse
import warnings
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from src.perception.models.teacher_bev import WaymoBEVDetector
from src.perception.models.student_bev import StudentBEVDetector
from src.perception.losses.distill_loss import CrossModalDistillationLoss
from src.perception.utils.dataset import WaymoDataset, waymo_collate_fn
from src.perception.utils.validate import validate_student
from src.perception.utils.target_encoder import BEVGridEncoder

# --- NEW: Exponential Moving Average (EMA) Helper ---
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in esd.keys():
                if esd[k].dtype.is_floating_point:
                    esd[k].mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
                else:
                    esd[k].copy_(msd[k])

def train_student(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Student Cross-Modal Distillation Training on: {device}")
    
    torch.backends.cudnn.benchmark = True
    writer = SummaryWriter(log_dir=args.log_dir)

    train_files = glob.glob('data/raw/train/*.tfrecord')
    val_files = glob.glob('data/raw/val/*.tfrecord')

    train_dataset = ConcatDataset([WaymoDataset(tfrecord_path=f, is_train=True) for f in train_files])
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=4,             # Reduced from 8 to fit in 16GB RAM
        pin_memory=True, 
        collate_fn=waymo_collate_fn, 
        persistent_workers=False,  # Clears worker memory leaks between epochs
        prefetch_factor=2
    )
    
    val_dataset = ConcatDataset([WaymoDataset(tfrecord_path=f, is_train=False) for f in val_files])
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=2,             # Reduced from 4
        pin_memory=True, 
        collate_fn=waymo_collate_fn, 
        persistent_workers=False,  # Clears memory after validation
        prefetch_factor=2
    )

    teacher = WaymoBEVDetector().to(device)
    teacher_checkpoint = args.teacher_ckpt
    if os.path.exists(teacher_checkpoint):
        print(f"Loading frozen Teacher model from {teacher_checkpoint}...")
        t_checkpoint = torch.load(teacher_checkpoint, map_location=device)
        teacher.load_state_dict(t_checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError("Teacher checkpoint not found!")

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    student = StudentBEVDetector().to(device)
    
    # Load Teacher's head weights into Student, then keep the head unfrozen so it can adapt
    student.bev_head.load_state_dict(teacher.bev_head.state_dict())
    for param in student.bev_head.parameters():
        param.requires_grad = True
    
    # Initialize EMA shadow model
    ema_student = ModelEMA(student, decay=0.999)
    
    # Initialize criteria with soft logit distillation
    criterion = CrossModalDistillationLoss(alpha_feat=0.1, alpha_logit=1.0, alpha_depth=0.1).to(device)
    encoder = BEVGridEncoder(x_range=(0.0, 70.0), y_range=(-40.0, 40.0), bev_h=160, bev_w=160)
    
    epochs = 15
    ACCUMULATION_STEPS = 8
    
    # Filter optimizer to include the SE block and the unfrozen head
    optimizer = optim.AdamW([
        {'params': filter(lambda p: p.requires_grad, student.backbone_stem.parameters()), 'lr': 1e-5},
        {'params': filter(lambda p: p.requires_grad, student.reduce_channel.parameters()), 'lr': 1e-4},
        {'params': filter(lambda p: p.requires_grad, student.view_transformer.parameters()), 'lr': 5e-4},
        {'params': filter(lambda p: p.requires_grad, student.adaptation.parameters()), 'lr': 5e-4},
        {'params': filter(lambda p: p.requires_grad, student.se_block.parameters()), 'lr': 5e-4},
        {'params': filter(lambda p: p.requires_grad, student.bev_head.parameters()), 'lr': 1e-4}
    ], weight_decay=1e-4)                                 
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    scaler = torch.amp.GradScaler('cuda')

    checkpoint_path = args.ckpt_name
    best_checkpoint_path = args.best_ckpt_name
    best_val_score = 0.0
    start_epoch = 0
    VAL_INTERVAL = 1

    if args.resume and os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        student.load_state_dict(checkpoint['model_state_dict'])
        ema_student.ema.load_state_dict(checkpoint.get('ema_state_dict', checkpoint['model_state_dict']))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_score = checkpoint.get('val_score', 0.0)
    
    for epoch in range(start_epoch, epochs):
        student.train() 
        
        epoch_train_loss = 0.0
        batch_start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(train_dataloader):
            camera_images = batch['camera_images'].to(device, non_blocking=True)
            lidar_points = batch['lidar_points'].to(device, non_blocking=True)
            batch_indices = batch['batch_indices'].to(device, non_blocking=True)
            lidar_uvs = batch['lidar_uvs'].to(device, non_blocking=True)
            
            intrinsics = batch['intrinsics'].to(device, non_blocking=True)
            extrinsics = batch['extrinsics'].to(device, non_blocking=True)
            
            extrinsics_inv = torch.inverse(extrinsics)
            depth_labels = batch['depth_labels'].to(device, non_blocking=True)
            
            encoded_targets = encoder.encode(batch['bboxes'], batch['num_valid_boxes'])
            targets_gpu = {k: v.to(device, non_blocking=True) for k, v in encoded_targets.items()}

            with torch.amp.autocast('cuda', dtype=torch.float16):
                with torch.no_grad():
                    teacher_outputs = teacher(lidar_points, batch_indices, camera_images, lidar_uvs)
                    
                    teacher_features = teacher_outputs.get('bev_features', None)
                    if teacher_features is not None:
                        teacher_features = torch.flip(teacher_features, dims=[-1])
                        
                    teacher_logits = teacher_outputs.get('bev_occupancy', None)
                    if teacher_logits is not None:
                        teacher_logits = torch.flip(teacher_logits, dims=[-1])

                student_outputs = student(camera_images, intrinsics, extrinsics_inv)
                
                total_loss, loss_dict = criterion(
                    student_outputs, 
                    teacher_features, 
                    ground_truth=targets_gpu['bev_occupancy'],
                    teacher_logits=teacher_logits,
                    depth_labels=depth_labels
                )
                loss = total_loss / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update EMA shadow weights after each optimizer step
                ema_student.update(student)
            
            true_loss = loss.item() * ACCUMULATION_STEPS
            epoch_train_loss += true_loss
            
            global_step = epoch * len(train_dataloader) + batch_idx
            writer.add_scalar('Training/Batch_Total_Loss', true_loss, global_step)
            writer.add_scalar('Training/Batch_Det_Loss', loss_dict['loss_det'], global_step)
            writer.add_scalar('Training/Batch_Feat_Loss', loss_dict['loss_feat'], global_step)
            writer.add_scalar('Training/Batch_Logit_Loss', loss_dict['loss_logit'], global_step)
            writer.add_scalar('Training/Batch_Depth_Loss', loss_dict['loss_depth'], global_step)
            
            if batch_idx % 10 == 0:
                elapsed_time = time.time() - batch_start_time
                batches_processed = 1 if batch_idx == 0 else 10
                sec_per_batch = elapsed_time / batches_processed
                print(f"Epoch {epoch + 1:02d}/{epochs} | Batch {batch_idx:03d} | Total Loss: {true_loss:.4f} | Depth Loss: {loss_dict['loss_depth']:.4f} | Speed: {sec_per_batch:.3f} s/b")
                batch_start_time = time.time()
                
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            scheduler.step()
        
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        
        if (epoch + 1) % VAL_INTERVAL == 0 or (epoch + 1) == epochs:
            avg_val_loss, val_score = validate_student(student, teacher, val_dataloader, criterion, encoder, device)
            print(f"Epoch {epoch + 1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | EMA Val Score: {val_score:.4f}")
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': student.state_dict(),
                'ema_state_dict': ema_student.ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_score': val_score,
            }
            
            if (epoch + 1) % 5 == 0:
                torch.save(checkpoint, checkpoint_path)
                print(f"--> Saved periodic checkpoint at epoch {epoch + 1} to {checkpoint_path}")
                
            if val_score > best_val_score:
                best_val_score = val_score
                torch.save(checkpoint, best_checkpoint_path)
                print(f"--> [NEW BEST] Saved peak EMA model with Val Score: {val_score:.4f} to {best_checkpoint_path}")

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Student Monocular BEV with EMA & SE')
    parser.add_argument('--resume', action='store_true', help='Resume training from best checkpoint')
    parser.add_argument('--teacher_ckpt', type=str, default='best_waymo_bev_checkpoint.pt', help='Path to frozen Teacher model')
    parser.add_argument('--log_dir', type=str, default='runs/camera_student_01', help='TensorBoard log directory')
    parser.add_argument('--ckpt_name', type=str, default='student_bev_checkpoint.pt', help='Standard checkpoint name')
    parser.add_argument('--best_ckpt_name', type=str, default='best_student_bev_checkpoint.pt', help='Best checkpoint name')
    args = parser.parse_args()
    
    mp.set_start_method('spawn', force=True)
    train_student(args)