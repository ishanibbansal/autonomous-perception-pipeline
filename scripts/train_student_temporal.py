import os
import sys
import copy
import glob
import time
import argparse
import warnings

# Inject root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from src.perception.models.teacher_bev import WaymoBEVDetector
from src.perception.models.student_bev import StudentBEVDetector
from src.perception.losses.distill_loss import CrossModalDistillationLoss
from src.perception.utils.dataset import WaymoTemporalDataset, temporal_collate_fn
from src.perception.utils.target_encoder import BEVGridEncoder

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

def validate_temporal(student, teacher, dataloader, criterion, encoder, device, seq_length=3):
    """Custom validation loop for the temporal sequence"""
    student.eval()
    val_loss_total = 0.0
    total_intersection = 0.0
    total_union = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            past_features = None
            past_extrinsics = None
            
            # Step forward through time (e.g., t_2 -> t_1 -> t_0)
            for step in reversed(range(seq_length)):
                step_key = f't_{step}'
                step_data = batch[step_key]
                
                cam = step_data['camera_images'].to(device, non_blocking=True)
                intrin = step_data['intrinsics'].to(device, non_blocking=True)
                extrin = step_data['extrinsics'].to(device, non_blocking=True)
                extrin_inv = torch.inverse(extrin)
                
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    s_out = student(
                        cam, intrin, extrin_inv, 
                        past_features=past_features, 
                        current_extrinsics=extrin, 
                        past_extrinsics=past_extrinsics
                    )
                
                past_features = s_out['bev_features']
                past_extrinsics = extrin
                
                if step == 0:
                    # Current frame evaluation
                    val_lidar = step_data['lidar_points'].to(device, non_blocking=True)
                    val_indices = step_data['batch_indices'].to(device, non_blocking=True)
                    val_uvs = step_data['lidar_uvs'].to(device, non_blocking=True)
                    val_depths = step_data['depth_labels'].to(device, non_blocking=True)
                    
                    encoded = encoder.encode(step_data['bboxes'], step_data['num_valid_boxes'])
                    val_gt = encoded['bev_occupancy'].to(device, non_blocking=True)

                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        t_out = teacher(val_lidar, val_indices, cam, val_uvs)
                        t_feat = t_out.get('bev_features', None)
                        if t_feat is not None: t_feat = torch.flip(t_feat, dims=[-1])

                        v_loss, v_dict = criterion(
                            s_out, t_feat, 
                            ground_truth=val_gt, 
                            teacher_logits=None, 
                            depth_labels=val_depths
                        )
                    
                    val_loss_total += v_dict['loss_total']
                    preds_binary = (torch.sigmoid(s_out['bev_occupancy']) > 0.15).float()
                    targets_binary = (val_gt > 0.3).float()
                    
                    intersection = (preds_binary * targets_binary).sum()
                    union = preds_binary.sum() + targets_binary.sum() - intersection
                    
                    total_intersection += intersection.item()
                    total_union += union.item()
                    
    return val_loss_total / len(dataloader), total_intersection / (total_union + 1e-6)

def train_student_temporal(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Phase 2: Memory-Safe Temporal Fusion Training on {device}")
    
    torch.backends.cudnn.benchmark = True
    writer = SummaryWriter(log_dir=args.log_dir)

    train_files = glob.glob('data/raw/train/*.tfrecord')
    val_files = glob.glob('data/raw/val/*.tfrecord')

    train_dataset = ConcatDataset([WaymoTemporalDataset(f, is_train=True, seq_length=args.seq_length) for f in train_files])
    train_dataloader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=4,           
        pin_memory=True, collate_fn=temporal_collate_fn, persistent_workers=True, prefetch_factor=1
    )
    
    val_dataset = ConcatDataset([WaymoTemporalDataset(f, is_train=False, seq_length=args.seq_length) for f in val_files])
    val_dataloader = DataLoader(
        val_dataset, batch_size=2, shuffle=False, num_workers=2,            
        pin_memory=True, collate_fn=temporal_collate_fn, persistent_workers=True, prefetch_factor=1
    )

    # 1. Load Frozen Teacher
    teacher = WaymoBEVDetector().to(device)
    t_checkpoint = torch.load(args.teacher_ckpt, map_location=device)
    teacher.load_state_dict(t_checkpoint['model_state_dict'])
    teacher.eval()
    for param in teacher.parameters(): param.requires_grad = False

    # 2. Load Student & Apply Phase 1 Spatial Weights
    student = StudentBEVDetector(use_temporal=True).to(device)
    print(f"Loading Phase 1 Spatial Backbone from {args.spatial_ckpt}...")
    s_checkpoint = torch.load(args.spatial_ckpt, map_location=device)
    student.load_state_dict(s_checkpoint['model_state_dict'], strict=False)

    # 3. FREEZE the Spatial Backbone to prevent catastrophic forgetting
    for param in student.backbone_stem.parameters(): param.requires_grad = False
    for param in student.reduce_channel.parameters(): param.requires_grad = False
    for param in student.view_transformer.parameters(): param.requires_grad = False
    for param in student.adaptation.parameters(): param.requires_grad = False
    for param in student.se_block.parameters(): param.requires_grad = False
    
    # 4. UNFREEZE Temporal Fusion and Head
    for param in student.temporal_fusion.parameters(): param.requires_grad = True
    for param in student.bev_head.parameters(): param.requires_grad = True

    ema_student = ModelEMA(student, decay=0.999)
    criterion = CrossModalDistillationLoss(alpha_feat=0.1, alpha_logit=1.0, alpha_depth=0.1).to(device)
    encoder = BEVGridEncoder(x_range=(0.0, 70.0), y_range=(-40.0, 40.0), bev_h=160, bev_w=160)
    
    epochs = 15
    ACCUMULATION_STEPS = 8
    
    # Optimizer specifically targets ONLY the active temporal parameters
    optimizer = optim.AdamW([
        {'params': filter(lambda p: p.requires_grad, student.temporal_fusion.parameters()), 'lr': 5e-4},
        {'params': filter(lambda p: p.requires_grad, student.bev_head.parameters()), 'lr': 1e-4}
    ], weight_decay=1e-4)                                 
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    scaler = torch.amp.GradScaler('cuda')

    best_val_score = 0.0
    
    for epoch in range(epochs):
        student.train() 
        epoch_train_loss = 0.0
        batch_start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(train_dataloader):
            past_features = None
            past_extrinsics = None
            
            # --- THE TIME MACHINE LOOP ---
            # Process chronological order: t_2 -> t_1 -> t_0
            for step in reversed(range(args.seq_length)):
                step_key = f't_{step}'
                step_data = batch[step_key]
                
                cam = step_data['camera_images'].to(device, non_blocking=True)
                intrin = step_data['intrinsics'].to(device, non_blocking=True)
                extrin = step_data['extrinsics'].to(device, non_blocking=True)
                extrin_inv = torch.inverse(extrin)
                
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    if step > 0:
                        # VRAM SHIELD: Process past frames WITHOUT tracking gradients
                        with torch.no_grad():
                            s_out = student(
                                cam, intrin, extrin_inv, 
                                past_features=past_features, 
                                current_extrinsics=extrin, 
                                past_extrinsics=past_extrinsics
                            )
                    else:
                        # CURRENT FRAME: Track gradients for the fusion block and head
                        s_out = student(
                            cam, intrin, extrin_inv, 
                            past_features=past_features, 
                            current_extrinsics=extrin, 
                            past_extrinsics=past_extrinsics
                        )
                
                # Cache for the next loop iteration
                past_features = s_out['bev_features']
                past_extrinsics = extrin
                
                # --- ONLY compute loss on the final Current Frame (t_0) ---
                if step == 0:
                    lidar_points = step_data['lidar_points'].to(device, non_blocking=True)
                    batch_indices = step_data['batch_indices'].to(device, non_blocking=True)
                    lidar_uvs = step_data['lidar_uvs'].to(device, non_blocking=True)
                    depth_labels = step_data['depth_labels'].to(device, non_blocking=True)
                    
                    encoded_targets = encoder.encode(step_data['bboxes'], step_data['num_valid_boxes'])
                    targets_gpu = {k: v.to(device, non_blocking=True) for k, v in encoded_targets.items()}

                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        with torch.no_grad():
                            t_out = teacher(lidar_points, batch_indices, cam, lidar_uvs)
                            t_feat = t_out.get('bev_features', None)
                            if t_feat is not None: t_feat = torch.flip(t_feat, dims=[-1])
                            t_logits = t_out.get('bev_occupancy', None)
                            if t_logits is not None: t_logits = torch.flip(t_logits, dims=[-1])

                        total_loss, loss_dict = criterion(
                            s_out, t_feat, ground_truth=targets_gpu['bev_occupancy'],
                            teacher_logits=t_logits, depth_labels=depth_labels
                        )
                        loss = total_loss / ACCUMULATION_STEPS
                    
                    scaler.scale(loss).backward()
            
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema_student.update(student)
            
            true_loss = loss.item() * ACCUMULATION_STEPS
            epoch_train_loss += true_loss
            
            if batch_idx % 10 == 0:
                sec_per_batch = (time.time() - batch_start_time) / (1 if batch_idx == 0 else 10)
                print(f"Epoch {epoch + 1:02d} | Batch {batch_idx:03d} | Total Loss: {true_loss:.4f} | Speed: {sec_per_batch:.3f} s/b")
                batch_start_time = time.time()
                
        scheduler.step()
        
        # Validation
        # Evaluate the active network directly to bypass EMA lag
        avg_val_loss, val_score = validate_temporal(student, teacher, val_dataloader, criterion, encoder, device, args.seq_length)
        print(f"Epoch {epoch + 1:02d}/{epochs} | Train Loss: {epoch_train_loss/len(train_dataloader):.4f} | EMA Val Score: {val_score:.4f}")
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': student.state_dict(),
            'ema_state_dict': ema_student.ema.state_dict(),
            'val_score': val_score,
        }
        
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(checkpoint, args.best_ckpt_name)
            print(f"--> [NEW BEST] Saved Temporal Model with Val Score: {val_score:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 2: Train Student Temporal Fusion')
    parser.add_argument('--teacher_ckpt', type=str, default='best_waymo_bev_checkpoint.pt')
    parser.add_argument('--spatial_ckpt', type=str, default='best_student_bev_checkpoint.pt', help='Phase 1 weights')
    parser.add_argument('--log_dir', type=str, default='runs/temporal_student_01')
    parser.add_argument('--best_ckpt_name', type=str, default='best_temporal_student_checkpoint.pt')
    parser.add_argument('--seq_length', type=int, default=3, help='Number of frames to process (t-2, t-1, t)')
    args = parser.parse_args()
    
    mp.set_start_method('spawn', force=True)
    train_student_temporal(args)