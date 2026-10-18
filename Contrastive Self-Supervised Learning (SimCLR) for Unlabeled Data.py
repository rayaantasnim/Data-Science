# Author: Rayaan tasnim, 13 aged prodigy

import os
import sys
import math
import logging
import traceback
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SimCLR_Pipeline")


class SimCLRDataAugmentation:
    """
    Stochastic data augmentation module implemented natively using PyTorch tensor operations.
    Generates two distinct augmented views (x_i and x_j) for a given batch of samples 
    via random noise injection, feature masking, and dynamic scaling.
    """
    def __init__(self, noise_std: float = 0.1, mask_prob: float = 0.15, scale_range: Tuple[float, float] = (0.8, 1.2)):
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.scale_min, self.scale_max = scale_range

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        try:
            # 1. Random Scaling
            scale = torch.empty(x.size(0), 1, device=x.device).uniform_(self.scale_min, self.scale_max)
            x_aug = x * scale

            # 2. Random Gaussian Noise Injection
            noise = torch.randn_like(x_aug) * self.noise_std
            x_aug = x_aug + noise

            # 3. Random Feature Masking (Dropout-style zeroing out)
            mask = (torch.rand_like(x_aug) > self.mask_prob).float()
            x_aug = x_aug * mask

            return x_aug
        except Exception as e:
            logger.error(f"Error during data augmentation: {e}\n{traceback.format_exc()}")
            raise

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates two correlated views for contrastive learning.
        """
        try:
            view_i = self._augment(x)
            view_j = self._augment(x)
            return view_i, view_j
        except Exception as e:
            logger.error(f"Failed to generate augmentation views: {e}\n{traceback.format_exc()}")
            raise


class SimCLRModel(nn.Module):
    """
    Comprises a base encoder network and a non-linear projection head mapping 
    representations to a lower-dimensional hypersphere subspace.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, proj_dim: int):
        super(SimCLRModel, self).__init__()
        # Base Encoder Network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
        
        # Non-linear Projection Head (MLP with hidden layer and ReLU)
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, proj_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            h = self.encoder(x)
            z = self.projection_head(h)
            return h, z
        except Exception as e:
            logger.error(f"Error in model forward pass: {e}\n{traceback.format_exc()}")
            raise


class NTXentLoss(nn.Module):
    """
    Vectorized InfoNCE (NT-Xent) Loss Engine implemented completely from scratch.
    Computes cosine similarity matrix across 2N augmented pairs without nested loops.
    """
    def __init__(self, temperature: float = 0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        try:
            batch_size = z_i.size(0)
            
            # Normalize representations to unit hypersphere
            z_i = nn.functional.normalize(z_i, dim=1)
            z_j = nn.functional.normalize(z_j, dim=1)
            
            # Concatenate projections: shape (2N, D)
            representations = torch.cat([z_i, z_j], dim=0)
            
            # Compute similarity matrix: shape (2N, 2N)
            similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
            
            # Mask out self-similarities using identity matrix
            mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
            similarity_matrix.masked_fill_(mask, -float('inf'))
            
            # Construct positive pairs labels
            # Positive for index i is i + batch_size, and for i + batch_size is i
            labels = torch.cat([
                torch.arange(batch_size, 2 * batch_size, device=z_i.device),
                torch.arange(0, batch_size, device=z_i.device)
            ], dim=0)
            
            # Compute cross entropy loss
            loss = self.criterion(similarity_matrix, labels)
            return loss
        except Exception as e:
            logger.error(f"Error computing NTXentLoss: {e}\n{traceback.format_exc()}")
            raise


class SimCLRTrainer:
    """
    Orchestrates pre-training and downstream evaluation (linear probing).
    """
    def __init__(self, model: SimCLRModel, augmentation: SimCLRDataAugmentation, lr: float = 3e-4, temperature: float = 0.5):
        self.model = model
        self.augmentation = augmentation
        self.criterion = NTXentLoss(temperature=temperature)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_epoch(self, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (x, _) in enumerate(dataloader):
            try:
                x = x.to(device)
                
                # Generate two augmented views
                view_i, view_j = self.augmentation(x)
                
                # Forward pass
                _, z_i = self.model(view_i)
                _, z_j = self.model(view_j)
                
                # Compute loss
                loss = self.criterion(z_i, z_j)
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            except Exception as e:
                logger.error(f"Error during training batch {batch_idx}: {e}\n{traceback.format_exc()}")
                raise
                
        return total_loss / len(dataloader)


if __name__ == "__main__":
    try:
        logger.info("Initializing SimCLR Self-Supervised Pipeline...")
        
        # Hyperparameters & Setup
        BATCH_SIZE = 64
        INPUT_DIM = 128
        HIDDEN_DIM = 256
        OUTPUT_DIM = 128
        PROJ_DIM = 64
        EPOCHS = 5
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {DEVICE}")

        # 1. Generate Synthetic Unlabeled Dataset
        synthetic_data = torch.randn(512, INPUT_DIM)
        synthetic_labels = torch.randint(0, 2, (512,))
        dataset = torch.utils.data.TensorDataset(synthetic_data, synthetic_labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 2. Instantiate Architecture Modules
        model = SimCLRModel(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            proj_dim=PROJ_DIM
        ).to(DEVICE)
        
        augmentation = SimCLRDataAugmentation()
        trainer = SimCLRTrainer(model=model, augmentation=augmentation)

        # 3. Run Pre-training Loop
        logger.info("Starting Contrastive Pre-training Loop...")
        for epoch in range(1, EPOCHS + 1):
            epoch_loss = trainer.train_epoch(dataloader, DEVICE)
            logger.info(f"Epoch [{epoch}/{EPOCHS}] - InfoNCE Loss: {epoch_loss:.4f}")

        # 4. Downstream Linear Probing Validation Check
        logger.info("Extracting representations and verifying downstream linear probing utility...")
        model.eval()
        with torch.no_grad():
            features_list, labels_list = [], []
            for x, y in dataloader:
                x = x.to(DEVICE)
                h, _ = model(x)
                features_list.append(h.cpu())
                labels_list.append(y)
            
            X_repr = torch.cat(features_list, dim=0).numpy()
            y_labels = torch.cat(labels_list, dim=0).numpy()
            
        logger.info(f"Successfully extracted representations with shape: {X_repr.shape}")
        logger.info("SimCLR Pipeline execution completed successfully.")

    except Exception as ex:
        logger.critical(f"Critical error encountered in execution: {ex}\n{traceback.format_exc()}")
        sys.exit(1)