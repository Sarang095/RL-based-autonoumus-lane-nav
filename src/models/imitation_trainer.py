"""
Imitation Learning Trainer for Vision-Based Autonomous Driving
Implements Behavioral Cloning with data augmentation and auxiliary tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
import pickle
import os
from tqdm import tqdm
import wandb
from .cnn_policy import CNNPolicy


class DrivingDataset(Dataset):
    """Dataset for storing driving demonstrations."""
    
    def __init__(
        self,
        observations: List[np.ndarray],
        actions: List[int],
        auxiliary_data: Optional[Dict[str, List]] = None,
        augment_data: bool = True,
        augmentation_prob: float = 0.5
    ):
        """
        Initialize the driving dataset.
        
        Args:
            observations: List of observation arrays
            actions: List of action indices
            auxiliary_data: Additional data like speed, lane position
            augment_data: Whether to apply data augmentation
            augmentation_prob: Probability of applying augmentation
        """
        self.observations = observations
        self.actions = actions
        self.auxiliary_data = auxiliary_data or {}
        self.augment_data = augment_data
        self.augmentation_prob = augmentation_prob
        
        assert len(observations) == len(actions), "Observations and actions must have same length"
        
    def __len__(self) -> int:
        return len(self.observations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        obs = self.observations[idx].copy()
        action = self.actions[idx]
        
        # Apply data augmentation
        if self.augment_data and np.random.random() < self.augmentation_prob:
            obs = self._augment_observation(obs)
            
        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs)
        action_tensor = torch.LongTensor([action])
        
        sample = {
            "observation": obs_tensor,
            "action": action_tensor
        }
        
        # Add auxiliary data if available
        for key, values in self.auxiliary_data.items():
            if idx < len(values):
                sample[key] = torch.FloatTensor([values[idx]])
                
        return sample
    
    def _augment_observation(self, obs: np.ndarray) -> np.ndarray:
        """Apply data augmentation to observation."""
        # Assuming obs shape is (C, H, W) where C might be stacked frames
        augmented = obs.copy()
        
        # Random brightness adjustment
        if np.random.random() < 0.3:
            brightness_factor = np.random.uniform(0.8, 1.2)
            augmented = np.clip(augmented * brightness_factor, 0, 1)
            
        # Random noise
        if np.random.random() < 0.2:
            noise = np.random.normal(0, 0.02, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 1)
            
        # Random horizontal flip (for some scenarios)
        if np.random.random() < 0.1:
            # Flip the spatial dimensions
            if len(augmented.shape) == 3:  # (C, H, W)
                augmented = augmented[:, :, ::-1]
            else:  # (H, W)
                augmented = augmented[:, ::-1]
                
        return augmented


class ImitationTrainer:
    """
    Trainer for imitation learning using behavioral cloning.
    """
    
    def __init__(
        self,
        policy: CNNPolicy,
        learning_rate: float = 1e-4,
        batch_size: int = 64,
        weight_decay: float = 1e-5,
        use_auxiliary_losses: bool = True,
        auxiliary_loss_weight: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the imitation trainer.
        
        Args:
            policy: The CNN policy to train
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            weight_decay: Weight decay for regularization
            use_auxiliary_losses: Whether to use auxiliary prediction tasks
            auxiliary_loss_weight: Weight for auxiliary losses
            device: Device to train on
        """
        self.policy = policy.to(device)
        self.device = device
        self.batch_size = batch_size
        self.use_auxiliary_losses = use_auxiliary_losses
        self.auxiliary_loss_weight = auxiliary_loss_weight
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Loss functions
        self.action_criterion = nn.CrossEntropyLoss()
        self.auxiliary_criterion = nn.MSELoss()
        
        # Training metrics
        self.training_metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "auxiliary_losses": []
        }
        
    def collect_demonstrations(
        self,
        env,
        expert_policy,
        num_episodes: int = 100,
        save_path: Optional[str] = None
    ) -> Tuple[List[np.ndarray], List[int], Dict[str, List]]:
        """
        Collect expert demonstrations from the environment.
        
        Args:
            env: The environment to collect demonstrations from
            expert_policy: Expert policy (can be scripted or human)
            num_episodes: Number of episodes to collect
            save_path: Path to save collected data
            
        Returns:
            Tuple of (observations, actions, auxiliary_data)
        """
        observations = []
        actions = []
        auxiliary_data = {"speeds": [], "lane_positions": []}
        
        print(f"Collecting {num_episodes} expert demonstrations...")
        
        for episode in tqdm(range(num_episodes)):
            obs, info = env.reset()
            done = False
            episode_obs = []
            episode_actions = []
            episode_speeds = []
            episode_lanes = []
            
            while not done:
                # Get expert action
                if hasattr(expert_policy, 'predict'):
                    # Stable-baselines3 style expert
                    action, _ = expert_policy.predict(obs, deterministic=True)
                elif callable(expert_policy):
                    # Custom expert function
                    action = expert_policy(obs, info)
                else:
                    # Random policy for testing
                    action = env.action_space.sample()
                
                # Store data
                episode_obs.append(obs)
                episode_actions.append(action)
                
                # Extract auxiliary information
                if hasattr(env, 'vehicle') and env.vehicle:
                    speed = getattr(env.vehicle, 'speed', 0.0)
                    lane_id = getattr(env.vehicle, 'lane_index', [None, None, 0])[2]
                    episode_speeds.append(speed / 30.0)  # Normalize speed
                    episode_lanes.append(lane_id)
                else:
                    episode_speeds.append(0.0)
                    episode_lanes.append(0)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
            # Only keep successful episodes (no collision)
            if not info.get('crashed', False) and len(episode_obs) > 10:
                observations.extend(episode_obs)
                actions.extend(episode_actions)
                auxiliary_data["speeds"].extend(episode_speeds)
                auxiliary_data["lane_positions"].extend(episode_lanes)
                
        print(f"Collected {len(observations)} demonstrations from {num_episodes} episodes")
        
        # Save data if path provided
        if save_path:
            data = {
                "observations": observations,
                "actions": actions,
                "auxiliary_data": auxiliary_data
            }
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved demonstrations to {save_path}")
            
        return observations, actions, auxiliary_data
    
    def load_demonstrations(self, load_path: str) -> Tuple[List[np.ndarray], List[int], Dict[str, List]]:
        """Load demonstrations from file."""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        return data["observations"], data["actions"], data["auxiliary_data"]
    
    def train(
        self,
        train_dataset: DrivingDataset,
        val_dataset: Optional[DrivingDataset] = None,
        num_epochs: int = 100,
        log_interval: int = 10,
        save_interval: int = 20,
        save_path: Optional[str] = None,
        use_wandb: bool = False
    ) -> Dict[str, List[float]]:
        """
        Train the policy using imitation learning.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of training epochs
            log_interval: Interval for logging metrics
            save_interval: Interval for saving checkpoints
            save_path: Path to save model checkpoints
            use_wandb: Whether to use Weights & Biases logging
            
        Returns:
            Dictionary of training metrics
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        best_val_loss = float('inf')
        
        print(f"Starting imitation learning training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, epoch)
            self.training_metrics["train_loss"].append(train_loss)
            self.training_metrics["train_accuracy"].append(train_acc)
            
            # Validation phase
            if val_loader:
                val_loss, val_acc = self._validate_epoch(val_loader, epoch)
                self.training_metrics["val_loss"].append(val_loss)
                self.training_metrics["val_accuracy"].append(val_acc)
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_path:
                        self._save_checkpoint(save_path, epoch, is_best=True)
            
            # Logging
            if epoch % log_interval == 0:
                log_msg = f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                if val_loader:
                    log_msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                print(log_msg)
                
                if use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss if val_loader else None,
                        "val_accuracy": val_acc if val_loader else None,
                        "learning_rate": self.optimizer.param_groups[0]['lr']
                    })
            
            # Save checkpoint
            if save_path and epoch % save_interval == 0:
                self._save_checkpoint(save_path, epoch)
                
        return self.training_metrics
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.policy.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            # Move data to device
            observations = batch["observation"].to(self.device)
            actions = batch["action"].to(self.device).squeeze()
            
            # Forward pass
            outputs = self.policy(observations, return_aux=self.use_auxiliary_losses)
            
            # Compute action loss
            action_logits = outputs["action_logits"]
            action_loss = self.action_criterion(action_logits, actions)
            
            total_loss_batch = action_loss
            
            # Compute auxiliary losses
            if self.use_auxiliary_losses:
                aux_loss = 0.0
                
                if "speeds" in batch and "speed_pred" in outputs:
                    speed_targets = batch["speeds"].to(self.device)
                    speed_loss = self.auxiliary_criterion(outputs["speed_pred"].squeeze(), speed_targets.squeeze())
                    aux_loss += speed_loss
                    
                if "lane_positions" in batch and "lane_pred" in outputs:
                    lane_targets = batch["lane_positions"].to(self.device).long()
                    lane_loss = self.action_criterion(outputs["lane_pred"], lane_targets.squeeze())
                    aux_loss += lane_loss
                    
                total_loss_batch += self.auxiliary_loss_weight * aux_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Compute accuracy
            with torch.no_grad():
                predicted = torch.argmax(action_logits, dim=1)
                accuracy = (predicted == actions).float().mean().item()
                
            total_loss += total_loss_batch.item()
            total_accuracy += accuracy
            num_batches += 1
            
        return total_loss / num_batches, total_accuracy / num_batches
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.policy.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                observations = batch["observation"].to(self.device)
                actions = batch["action"].to(self.device).squeeze()
                
                # Forward pass
                outputs = self.policy(observations, return_aux=self.use_auxiliary_losses)
                
                # Compute losses
                action_logits = outputs["action_logits"]
                loss = self.action_criterion(action_logits, actions)
                
                # Compute accuracy
                predicted = torch.argmax(action_logits, dim=1)
                accuracy = (predicted == actions).float().mean().item()
                
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
                
        return total_loss / num_batches, total_accuracy / num_batches
    
    def _save_checkpoint(self, save_path: str, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_metrics": self.training_metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = f"{save_path}_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = f"{save_path}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.training_metrics = checkpoint["training_metrics"]
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint["epoch"]