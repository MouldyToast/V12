"""
Singular Space Residual Diffusion Training - Version 1

VERSION 1 IMPLEMENTATION
========================
Changes from original:
1. Loads original_lengths from preprocessed data (for validation/analysis)
2. Supports variable K (default K=40 for Version 1)
3. Reports length statistics during training
4. Compatible with B-spline preprocessed data

This script trains a diffusion model to predict residuals from group anchors
in the K-dimensional Singular space.

Architecture:
    - Input: noisy residual r_t, timestep t, orientation_id, distance_id, anchor
    - Output: predicted noise ε_θ
    - The model learns to denoise residuals, NOT full trajectories

Key Innovation (from SingularTrajectory):
    - Start from anchor prototypes, not Gaussian noise
    - Predict small residuals (what diffusion must learn is minimal)
    - Only M=10 diffusion steps needed

Usage:

    ############# ORIGINAL VERSION  ##########
    python train_singular_diffusion_v1.py --data processed_v3/ --epochs 256


    # Resume training
    python train_singular_diffusion_v1.py --data processed_singular/ --resume checkpoints/latest.pt
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
import math
import json
from datetime import datetime
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Model
    'hidden_dim': 256,
    'num_layers': 4,
    'dropout': 0.1,
    
    # Diffusion
    'diffusion_steps': 2, #WAS 10      # M (small because anchor-based!)
    'beta_start': 0.0001,
    'beta_end': 0.02,
    
    # Training
    'batch_size': 128,
    'learning_rate': 1e-3,# 3e-4
    'weight_decay': 1e-4,
    'epochs': 256,
    'warmup_epochs': 10,
    
    # Logging
    'log_every': 100,
    'save_every': 2000,
    'validate_every': 1,
}


# =============================================================================
# DATASET - Version 1 with original_lengths support
# =============================================================================

class SingularTrajectoryDataset:
    """
    Dataset that lives entirely on GPU - eliminates data loading bottleneck.
    
    VERSION 1 CHANGES:
    - Loads original_lengths if present (for validation/analysis)
    - Reports length statistics
    """
    
    def __init__(self, data_dir, split='train', device='cuda'):
        self.data_dir = Path(data_dir)
        self.split = split
        self.device = device
        
        split_dir = self.data_dir / split
        
        # Load everything to GPU immediately
        self.residuals = torch.tensor(
            np.load(split_dir / 'residuals.npy'),
            dtype=torch.float32, device=device
        )
        self.coefficients = torch.tensor(
            np.load(split_dir / 'coefficients.npy'),
            dtype=torch.float32, device=device
        )
        self.anchor_indices = torch.tensor(
            np.load(split_dir / 'anchor_indices.npy'),
            dtype=torch.long, device=device
        )
        self.orientation_ids = torch.tensor(
            np.load(split_dir / 'orientation_ids.npy'),
            dtype=torch.long, device=device
        )
        self.distance_ids = torch.tensor(
            np.load(split_dir / 'distance_ids.npy'),
            dtype=torch.long, device=device
        )
        
        # VERSION 1: Load original_lengths if present
        original_lengths_path = split_dir / 'original_lengths.npy'
        if original_lengths_path.exists():
            self.original_lengths = torch.tensor(
                np.load(original_lengths_path),
                dtype=torch.long, device=device
            )
            self.has_lengths = True
        else:
            self.original_lengths = None
            self.has_lengths = False
        
        # Load config and anchors
        self.config = np.load(self.data_dir / 'config.npy', allow_pickle=True).item()
        self.group_anchors = np.load(self.data_dir / 'group_anchors.npy', allow_pickle=True).item()
        
        self.K = self.config['K']
        self.T_win = self.config['T_win']
        self.num_samples = len(self.residuals)
        
        # Precompute anchors for each sample (on GPU)
        self.anchors = self._precompute_anchors()
        
        # Report statistics
        print(f"  Preloaded {split}: {self.num_samples} samples to {device}")
        print(f"    K={self.K}, T_win={self.T_win}")
        if self.has_lengths:
            lengths_cpu = self.original_lengths.cpu().numpy()
            print(f"    Original lengths: min={lengths_cpu.min()}, max={lengths_cpu.max()}, "
                  f"median={np.median(lengths_cpu):.0f}")
    
    def _precompute_anchors(self):
        """Precompute anchor for each sample."""
        anchors = np.zeros((self.num_samples, self.K), dtype=np.float32)

        # Need to access CPU values for dict lookup
        dist_ids_cpu = self.distance_ids.cpu().numpy()
        orient_ids_cpu = self.orientation_ids.cpu().numpy()
        anchor_indices_cpu = self.anchor_indices.cpu().numpy()
        
        for i in range(self.num_samples):
            key = (int(dist_ids_cpu[i]), int(orient_ids_cpu[i]))
            anchor_idx = int(anchor_indices_cpu[i])
            
            group_anchors = self.group_anchors.get(key)
            if group_anchors is not None and anchor_idx >= 0:
                anchors[i] = group_anchors[anchor_idx]
        
        return torch.tensor(anchors, dtype=torch.float32, device=self.device)
    
    def __len__(self):
        return self.num_samples
    
    def get_batch(self, indices):
        """Get a batch by indices - everything already on GPU!"""
        batch = {
            'residual': self.residuals[indices],
            'anchor': self.anchors[indices],
            'orientation_id': self.orientation_ids[indices],
            'distance_id': self.distance_ids[indices],
            'coefficient': self.coefficients[indices],
        }
        
        # Include original_lengths if available
        if self.has_lengths:
            batch['original_length'] = self.original_lengths[indices]
        
        return batch


# =============================================================================
# EMBEDDINGS
# =============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for diffusion timestep."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


# =============================================================================
# DIFFUSION MODEL
# =============================================================================

class ResidualDiffusionMLP(nn.Module):
    """
    MLP-based diffusion model for residual prediction.
    
    Predicts noise ε given:
        - r_t: noisy residual at timestep t
        - t: diffusion timestep
        - orient_id: orientation class (0-7)
        - dist_id: distance group class (0-4)
        - anchor: selected anchor prototype
    
    VERSION 1 NOTE:
    - K can be any value (default 40 for Version 1)
    - Input dimension scales with K: 2*K + 64 + 64 + 128
    """
    
    def __init__(self, K=40, hidden_dim=256, num_layers=4, 
                 num_orientations=8, num_distances=5, dropout=0.1):
        super().__init__()
        
        self.K = K
        self.hidden_dim = hidden_dim
        
        # Embeddings
        self.orient_embed = nn.Embedding(num_orientations, 64)
        self.dist_embed = nn.Embedding(num_distances, 64)
        self.time_embed = SinusoidalPositionEmbeddings(128)
        
        # Input projection
        # Input: r_t (K) + anchor (K) + orient_emb (64) + dist_emb (64) + time_emb (128)
        input_dim = K + K + 64 + 64 + 128
        
        # MLP layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, K))  # Output: predicted noise
        
        self.net = nn.Sequential(*layers)
        
        # Initialize output layer to small values
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=0.01)
    
    def forward(self, r_t, t, orient_id, dist_id, anchor):
        """
        Forward pass.
        
        Args:
            r_t: [B, K] noisy residual
            t: [B] timestep (integer 0 to M-1)
            orient_id: [B] orientation class
            dist_id: [B] distance class
            anchor: [B, K] anchor prototype
        
        Returns:
            noise_pred: [B, K] predicted noise
        """
        # Get embeddings
        orient_emb = self.orient_embed(orient_id)  # [B, 64]
        dist_emb = self.dist_embed(dist_id)        # [B, 64]
        time_emb = self.time_embed(t.float())      # [B, 128]
        
        # Concatenate all inputs
        x = torch.cat([r_t, anchor, orient_emb, dist_emb, time_emb], dim=-1)
        
        # Predict noise
        noise_pred = self.net(x)
        
        return noise_pred


# =============================================================================
# NOISE SCHEDULE
# =============================================================================

class NoiseSchedule:
    """Linear noise schedule for diffusion."""
    
    def __init__(self, num_steps, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_steps = num_steps
        self.device = device
        
        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # For sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
    def add_noise(self, x_0, t, noise=None):
        """
        Forward diffusion: add noise to x_0.
        
        q(x_t | x_0) = N(x_t; sqrt(ᾱ_t) * x_0, (1-ᾱ_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise
    
    def ddim_step(self, x_t, noise_pred, t, eta=0.0):
        """
        DDIM sampling step (deterministic when eta=0).
        """
        alpha_t = self.alphas_cumprod[t].view(-1, 1)
        alpha_prev = self.alphas_cumprod_prev[t].view(-1, 1)
        
        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        
        # Direction pointing to x_t
        direction = torch.sqrt(1 - alpha_prev) * noise_pred
        
        # DDIM update
        x_prev = torch.sqrt(alpha_prev) * x_0_pred + direction
        
        return x_prev


# =============================================================================
# TRAINING
# =============================================================================

class Trainer:
    """Training loop for residual diffusion model."""
    
    def __init__(self, model, train_dataset, val_dataset, config, device='cuda'):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Calculate batches per epoch
        batch_size = config['batch_size']
        batches_per_epoch = (len(train_dataset) + batch_size - 1) // batch_size

        # Learning rate scheduler (cosine with warmup)
        total_steps = config['epochs'] * batches_per_epoch
        warmup_steps = config['warmup_epochs'] * batches_per_epoch
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Noise schedule
        self.noise_schedule = NoiseSchedule(
            config['diffusion_steps'],
            config['beta_start'],
            config['beta_end'],
            device=device
        )
        
        # Logging
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        # Create random permutation for shuffling
        num_samples = len(self.train_dataset)
        indices = torch.randperm(num_samples, device=self.device)
        batch_size = self.config['batch_size']
    
        # Calculate number of batches
        n_batches = (num_samples + batch_size - 1) // batch_size
        
        pbar = tqdm(range(n_batches), desc=f'Epoch {epoch}')
        for batch_idx in pbar:

            # Get batch indices
            start = batch_idx * batch_size
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            # Get batch (already on GPU!)
            batch = self.train_dataset.get_batch(batch_indices)

            residuals = batch['residual']
            anchors = batch['anchor']
            orient_ids = batch['orientation_id']
            dist_ids = batch['distance_id']
            
            B = residuals.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, self.config['diffusion_steps'], (B,), device=self.device)
            
            # Add noise to residuals
            r_t, noise = self.noise_schedule.add_noise(residuals, t)
            
            # Predict noise
            noise_pred = self.model(r_t, t, orient_ids, dist_ids, anchors)
            
            # Loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        
        avg_loss = total_loss / num_batches
        self.history['train_loss'].append(avg_loss)
        self.history['lr'].append(self.scheduler.get_last_lr()[0])
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Run validation."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        num_samples = len(self.val_dataset)
        batch_size = self.config['batch_size']
        indices = torch.arange(num_samples, device=self.device)

        n_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            batch = self.val_dataset.get_batch(batch_indices)

            residuals = batch['residual']
            anchors = batch['anchor']
            orient_ids = batch['orientation_id']
            dist_ids = batch['distance_id']
            
            B = residuals.shape[0]
            
            # Use all timesteps for validation
            for t_val in range(self.config['diffusion_steps']):
                t = torch.full((B,), t_val, device=self.device)
                r_t, noise = self.noise_schedule.add_noise(residuals, t)
                noise_pred = self.model(r_t, t, orient_ids, dist_ids, anchors)
                loss = F.mse_loss(noise_pred, noise)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.history['val_loss'].append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, path, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config,
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = Path(path).parent / 'best_model.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        return checkpoint['epoch']


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Singular space residual diffusion (Version 1)')
    
    parser.add_argument('--data', type=str, default='processed_singular',
                        help='Path to preprocessed data directory')
    parser.add_argument('--output', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()

    # =========================================================================
    # GPU SPEED OPTIMIZATIONS
    # =========================================================================
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("TF32 + cuDNN benchmark enabled")
    
    print("="*70)
    print("SINGULAR SPACE RESIDUAL DIFFUSION TRAINING (VERSION 1)")
    print("="*70)
    print(f"Data: {args.data}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print("="*70)
    
    # Load config from preprocessed data
    data_config = np.load(Path(args.data) / 'config.npy', allow_pickle=True).item()
    K = data_config['K']
    T_win = data_config['T_win']
    
    # Check for Version 1 markers
    version = data_config.get('version', 'unknown')
    print(f"\nData config: K={K}, T_win={T_win}, version={version}")
    
    # Build config
    config = DEFAULT_CONFIG.copy()
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['K'] = K
    config['T_win'] = T_win  # Store T_win in checkpoint for generation
    
    # =========================================================================
    # GPU PRELOADING: Load entire dataset to GPU
    # =========================================================================
    print("\nPreloading datasets to GPU...")
    train_dataset = SingularTrajectoryDataset(args.data, 'train', device=args.device)
    val_dataset = SingularTrajectoryDataset(args.data, 'val', device=args.device)
    
    print(f"\nTrain: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Batches per epoch: {(len(train_dataset) + config['batch_size'] - 1) // config['batch_size']}")

    # Create model
    print("\nCreating model...")
    model = ResidualDiffusionMLP(
        K=K,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_orientations=data_config['num_orientations'],
        num_distances=data_config['num_distance_groups'],
        dropout=config['dropout']
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Version 1 with K=40: estimate parameter count
    # Input: 40 + 40 + 64 + 64 + 128 = 336
    # Layer 1: 336 * 256 + 256 = 86,272
    # Layer 2: 256 * 256 + 256 = 65,792
    # Layer 3: 256 * 256 + 256 = 65,792
    # Layer 4: 256 * 40 + 40 = 10,280
    # Embeddings: 8*64 + 5*64 = 832
    # Total: ~229K parameters
    
    # Create trainer
    trainer = Trainer(model, train_dataset, val_dataset, config, device=args.device)
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume) + 1
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    for epoch in range(start_epoch, config['epochs']):
        # Train
        train_loss = trainer.train_epoch(epoch)
        
        # Validate
        if epoch % config['validate_every'] == 0:
            val_loss = trainer.validate()
            
            print(f"\nEpoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
            # Save checkpoint
            is_best = val_loss < trainer.best_val_loss
            if is_best:
                trainer.best_val_loss = val_loss
                print(f"  New best model! (val_loss={val_loss:.6f})")
            
            if epoch % config['save_every'] == 0 or is_best:
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
                trainer.save_checkpoint(checkpoint_path, epoch, is_best)
                print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save latest
        trainer.save_checkpoint(output_dir / 'latest.pt', epoch)
    
    # Save final model
    trainer.save_checkpoint(output_dir / 'final_model.pt', config['epochs'] - 1)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE (VERSION 1)")
    print("="*70)
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"\nNext step: Generate trajectories with:")
    print(f"  python generate_trajectory_v1.py --checkpoint {output_dir}/best_model.pt --data {args.data}")


if __name__ == '__main__':
    main()
