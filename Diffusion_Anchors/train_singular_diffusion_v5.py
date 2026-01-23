"""
Singular Space Residual Diffusion Training - Version 5 (Anticipatory)

V5 INNOVATION: Dual Conditioning (AB + BC)
==========================================
The model learns trajectory SHAPE based on:
    - AB: Current segment direction (where we're going)
    - BC: Next segment direction (where we're going AFTER)
    - Turn category: How much direction changes

This captures ANTICIPATORY MOVEMENT - how humans naturally curve
toward the next target before reaching the current one.

Architecture:
    Input: r_t, timestep, orient_AB, dist_AB, orient_BC, dist_BC, turn_cat, anchor
    Output: predicted noise ε_θ

Conditioning dimensions:
    - orient_AB (0-7) → 64-dim embedding
    - dist_AB (0-4) → 64-dim embedding
    - orient_BC (0-7) → 64-dim embedding
    - dist_BC (0-4) → 64-dim embedding
    - turn_category (0-6) → 64-dim embedding
    - time → 128-dim sinusoidal embedding
    Total conditioning: 320 + 128 = 448 dims

Anchors: Per turn-category (7 categories)

Usage:
    python train_singular_diffusion_v5.py --data processed_v5/ --epochs 256

    # Resume training
    python train_singular_diffusion_v5.py --data processed_v5/ --resume checkpoints/latest.pt
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    'diffusion_steps': 10,
    'beta_start': 0.0001,
    'beta_end': 0.02,

    # Training
    'batch_size': 128,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 256,
    'warmup_epochs': 20,

    # Logging
    'log_every': 100,
    'save_every': 10,
    'validate_every': 1,
}


# =============================================================================
# DATASET - V5 with AB, BC, Turn conditioning (GPU preloaded)
# =============================================================================

class SingularTrajectoryDatasetV5:
    """
    Dataset for V5 anticipatory conditioning.
    Preloads everything to GPU for maximum training speed.

    Loads:
        - residuals, coefficients, anchor_indices
        - orientation_ids_AB, distance_ids_AB (current segment)
        - orientation_ids_BC, distance_ids_BC (next segment)
        - turn_category_ids (for anchor selection)
        - original_lengths
    """

    def __init__(self, data_dir, split='train', device='cuda'):
        self.data_dir = Path(data_dir)
        self.split = split
        self.device = device

        split_dir = self.data_dir / split

        # Load core arrays to GPU
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

        # V5: Load AB conditioning (current segment)
        self.orientation_ids_AB = torch.tensor(
            np.load(split_dir / 'orientation_ids_AB.npy'),
            dtype=torch.long, device=device
        )
        self.distance_ids_AB = torch.tensor(
            np.load(split_dir / 'distance_ids_AB.npy'),
            dtype=torch.long, device=device
        )

        # V5: Load BC conditioning (next segment - anticipation!)
        self.orientation_ids_BC = torch.tensor(
            np.load(split_dir / 'orientation_ids_BC.npy'),
            dtype=torch.long, device=device
        )
        self.distance_ids_BC = torch.tensor(
            np.load(split_dir / 'distance_ids_BC.npy'),
            dtype=torch.long, device=device
        )

        # V5: Load turn category
        self.turn_category_ids = torch.tensor(
            np.load(split_dir / 'turn_category_ids.npy'),
            dtype=torch.long, device=device
        )

        # Load original lengths if present
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

        # Load config and turn-category anchors
        self.config = np.load(self.data_dir / 'config.npy', allow_pickle=True).item()
        self.turn_anchors = np.load(
            self.data_dir / 'turn_category_anchors.npy',
            allow_pickle=True
        ).item()

        self.K = self.config['K']
        self.T_ref = self.config.get('T_ref', self.config.get('T_win', 30))
        self.num_samples = len(self.residuals)

        # Precompute anchors for each sample
        self.anchors = self._precompute_anchors()

        # Report statistics
        print(f"  Preloaded {split}: {self.num_samples} samples to {device}")
        print(f"    K={self.K}, T_ref={self.T_ref}")

        if self.has_lengths:
            lengths_cpu = self.original_lengths.cpu().numpy()
            print(f"    Lengths: min={lengths_cpu.min()}, max={lengths_cpu.max()}, "
                  f"median={np.median(lengths_cpu):.0f}")

        # Turn category distribution
        turn_cats_cpu = self.turn_category_ids.cpu().numpy()
        turn_counts = np.bincount(turn_cats_cpu, minlength=7)
        print(f"    Turn categories: {turn_counts.tolist()}")

    def _precompute_anchors(self):
        """Precompute anchor for each sample based on turn category."""
        anchors = np.zeros((self.num_samples, self.K), dtype=np.float32)

        turn_cats_cpu = self.turn_category_ids.cpu().numpy()
        anchor_indices_cpu = self.anchor_indices.cpu().numpy()

        for i in range(self.num_samples):
            turn_cat = int(turn_cats_cpu[i])
            anchor_idx = int(anchor_indices_cpu[i])

            cat_anchors = self.turn_anchors.get(turn_cat)
            if cat_anchors is not None and anchor_idx >= 0 and anchor_idx < len(cat_anchors):
                anchors[i] = cat_anchors[anchor_idx]

        return torch.tensor(anchors, dtype=torch.float32, device=self.device)

    def __len__(self):
        return self.num_samples

    def get_batch(self, indices):
        """Get a batch by indices - everything already on GPU."""
        return {
            'residual': self.residuals[indices],
            'anchor': self.anchors[indices],
            'coefficient': self.coefficients[indices],

            # AB conditioning (current segment)
            'orientation_id_AB': self.orientation_ids_AB[indices],
            'distance_id_AB': self.distance_ids_AB[indices],

            # BC conditioning (next segment)
            'orientation_id_BC': self.orientation_ids_BC[indices],
            'distance_id_BC': self.distance_ids_BC[indices],

            # Turn category
            'turn_category_id': self.turn_category_ids[indices],
        }


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
# V5 DIFFUSION MODEL - Dual Conditioning
# =============================================================================

class ResidualDiffusionMLP_V5(nn.Module):
    """
    MLP diffusion model with V5 dual conditioning.

    Predicts noise ε given:
        - r_t: noisy residual at timestep t
        - t: diffusion timestep
        - orient_AB, dist_AB: current segment direction
        - orient_BC, dist_BC: next segment direction (anticipation!)
        - turn_cat: turn category (0-6)
        - anchor: selected anchor prototype

    Input dimension: 2*K + 64*5 + 128 = 2*K + 448
    """

    def __init__(self, K=15, hidden_dim=256, num_layers=4,
                 num_orientations=8, num_distances=5, num_turn_categories=7,
                 dropout=0.1):
        super().__init__()

        self.K = K
        self.hidden_dim = hidden_dim

        # AB embeddings (current segment)
        self.orient_AB_embed = nn.Embedding(num_orientations, 64)
        self.dist_AB_embed = nn.Embedding(num_distances, 64)

        # BC embeddings (next segment - anticipation!)
        self.orient_BC_embed = nn.Embedding(num_orientations, 64)
        self.dist_BC_embed = nn.Embedding(num_distances, 64)

        # Turn category embedding
        self.turn_cat_embed = nn.Embedding(num_turn_categories, 64)

        # Time embedding
        self.time_embed = SinusoidalPositionEmbeddings(128)

        # Input projection
        # r_t (K) + anchor (K) + AB (128) + BC (128) + turn (64) + time (128)
        input_dim = K + K + 64 + 64 + 64 + 64 + 64 + 128  # = 2K + 448

        # MLP layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, K))

        self.net = nn.Sequential(*layers)

        # Initialize output layer small
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(self, r_t, t, orient_AB, dist_AB, orient_BC, dist_BC, turn_cat, anchor):
        """
        Forward pass with dual conditioning.

        Args:
            r_t: [B, K] noisy residual
            t: [B] timestep
            orient_AB: [B] current segment orientation (0-7)
            dist_AB: [B] current segment distance group (0-4)
            orient_BC: [B] next segment orientation (0-7)
            dist_BC: [B] next segment distance group (0-4)
            turn_cat: [B] turn category (0-6)
            anchor: [B, K] anchor prototype

        Returns:
            noise_pred: [B, K] predicted noise
        """
        # AB embeddings (current segment)
        orient_AB_emb = self.orient_AB_embed(orient_AB)  # [B, 64]
        dist_AB_emb = self.dist_AB_embed(dist_AB)        # [B, 64]

        # BC embeddings (next segment - anticipation!)
        orient_BC_emb = self.orient_BC_embed(orient_BC)  # [B, 64]
        dist_BC_emb = self.dist_BC_embed(dist_BC)        # [B, 64]

        # Turn category embedding
        turn_emb = self.turn_cat_embed(turn_cat)         # [B, 64]

        # Time embedding
        time_emb = self.time_embed(t.float())            # [B, 128]

        # Concatenate all inputs
        x = torch.cat([
            r_t,
            anchor,
            orient_AB_emb,
            dist_AB_emb,
            orient_BC_emb,
            dist_BC_emb,
            turn_emb,
            time_emb
        ], dim=-1)

        # Predict noise
        return self.net(x)


# =============================================================================
# NOISE SCHEDULE
# =============================================================================

class NoiseSchedule:
    """Linear noise schedule for diffusion."""

    def __init__(self, num_steps, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_steps = num_steps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def add_noise(self, x_0, t, noise=None):
        """Forward diffusion: add noise to x_0."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    def ddim_step(self, x_t, noise_pred, t, eta=0.0):
        """DDIM sampling step."""
        alpha_t = self.alphas_cumprod[t].view(-1, 1)
        alpha_prev = self.alphas_cumprod_prev[t].view(-1, 1)

        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        direction = torch.sqrt(1 - alpha_prev) * noise_pred
        x_prev = torch.sqrt(alpha_prev) * x_0_pred + direction

        return x_prev


# =============================================================================
# TRAINER
# =============================================================================

class TrainerV5:
    """Training loop for V5 residual diffusion model."""

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

        # LR scheduler
        batch_size = config['batch_size']
        batches_per_epoch = (len(train_dataset) + batch_size - 1) // batch_size
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

        num_samples = len(self.train_dataset)
        indices = torch.randperm(num_samples, device=self.device)
        batch_size = self.config['batch_size']
        n_batches = (num_samples + batch_size - 1) // batch_size

        pbar = tqdm(range(n_batches), desc=f'Epoch {epoch}')
        for batch_idx in pbar:
            start = batch_idx * batch_size
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            batch = self.train_dataset.get_batch(batch_indices)

            residuals = batch['residual']
            anchors = batch['anchor']
            orient_AB = batch['orientation_id_AB']
            dist_AB = batch['distance_id_AB']
            orient_BC = batch['orientation_id_BC']
            dist_BC = batch['distance_id_BC']
            turn_cat = batch['turn_category_id']

            B = residuals.shape[0]

            # Sample random timesteps
            t = torch.randint(0, self.config['diffusion_steps'], (B,), device=self.device)

            # Add noise
            r_t, noise = self.noise_schedule.add_noise(residuals, t)

            # Predict noise with V5 dual conditioning
            noise_pred = self.model(
                r_t, t,
                orient_AB, dist_AB,
                orient_BC, dist_BC,
                turn_cat,
                anchors
            )

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

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
            orient_AB = batch['orientation_id_AB']
            dist_AB = batch['distance_id_AB']
            orient_BC = batch['orientation_id_BC']
            dist_BC = batch['distance_id_BC']
            turn_cat = batch['turn_category_id']

            B = residuals.shape[0]

            # Test all timesteps
            for t_val in range(self.config['diffusion_steps']):
                t = torch.full((B,), t_val, device=self.device)
                r_t, noise = self.noise_schedule.add_noise(residuals, t)

                noise_pred = self.model(
                    r_t, t,
                    orient_AB, dist_AB,
                    orient_BC, dist_BC,
                    turn_cat,
                    anchors
                )

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
            'version': 'v5_anticipatory',
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
    parser = argparse.ArgumentParser(
        description='Train V5 anticipatory diffusion model'
    )

    parser.add_argument('--data', type=str, default='processed_v5',
                        help='Path to preprocessed data')
    parser.add_argument('--output', type=str, default='checkpoints_v5',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['learning_rate'])
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("TF32 + cuDNN benchmark enabled")

    print("=" * 70)
    print("V5 ANTICIPATORY DIFFUSION TRAINING")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print("=" * 70)

    # Load data config
    data_config = np.load(Path(args.data) / 'config.npy', allow_pickle=True).item()
    K = data_config['K']
    T_ref = data_config.get('T_ref', data_config.get('T_win', 30))
    version = data_config.get('version', 'unknown')

    print(f"\nData: K={K}, T_ref={T_ref}, version={version}")

    # Build config
    config = DEFAULT_CONFIG.copy()
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['K'] = K
    config['T_ref'] = T_ref

    # Load datasets to GPU
    print("\nPreloading datasets to GPU...")
    train_dataset = SingularTrajectoryDatasetV5(args.data, 'train', device=args.device)
    val_dataset = SingularTrajectoryDatasetV5(args.data, 'val', device=args.device)

    print(f"\nTrain: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

    # Create model
    print("\nCreating V5 model...")
    model = ResidualDiffusionMLP_V5(
        K=K,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_orientations=data_config.get('num_orientations', 8),
        num_distances=data_config.get('num_distance_groups', 5),
        num_turn_categories=data_config.get('num_turn_categories', 7),
        dropout=config['dropout']
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = TrainerV5(model, train_dataset, val_dataset, config, device=args.device)

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
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(start_epoch, config['epochs']):
        train_loss = trainer.train_epoch(epoch)

        if epoch % config['validate_every'] == 0:
            val_loss = trainer.validate()

            print(f"\nEpoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}")

            is_best = val_loss < trainer.best_val_loss
            if is_best:
                trainer.best_val_loss = val_loss
                print(f"  New best! (val={val_loss:.6f})")

            if epoch % config['save_every'] == 0 or is_best:
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
                trainer.save_checkpoint(checkpoint_path, epoch, is_best)
                print(f"  Saved: {checkpoint_path}")

        trainer.save_checkpoint(output_dir / 'latest.pt', epoch)

    # Final save
    trainer.save_checkpoint(output_dir / 'final_model.pt', config['epochs'] - 1)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE (V5 ANTICIPATORY)")
    print("=" * 70)
    print(f"Best val loss: {trainer.best_val_loss:.6f}")
    print(f"Checkpoints: {output_dir}")
    print(f"\nGenerate trajectories with:")
    print(f"  python generate_trajectory_v5.py --checkpoint {output_dir}/best_model.pt --data {args.data}")


if __name__ == '__main__':
    main()
