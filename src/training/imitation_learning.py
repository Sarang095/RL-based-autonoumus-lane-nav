"""
Imitation Learning (Behavioral Cloning) Training Module
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..models import create_model
from ..data_collection import create_expert_dataset
from ..config import Config
from ..environment_wrapper import create_environment


class ImitationLearner:
    """
    Imitation Learning trainer using Behavioral Cloning
    """
    
    def __init__(self, 
                 env_name: str = 'highway',
                 model_config: Dict[str, Any] = None,
                 training_config: Dict[str, Any] = None):
        
        self.env_name = env_name
        self.model_config = model_config or Config.CNN_CONFIG
        self.training_config = training_config or Config.IL_CONFIG
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = create_model('cnn', self.model_config)
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.training_config['learning_rate']
        )
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        
        # Setup logging
        self.setup_logging()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def setup_logging(self):
        """Setup tensorboard logging"""
        log_dir = os.path.join(Config.PATHS['logs_dir'], 'imitation_learning')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
    def prepare_data(self, num_episodes: int = None, force_recollect: bool = False) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders
        
        Args:
            num_episodes: Number of expert episodes to collect
            force_recollect: Whether to force recollection of expert data
            
        Returns:
            train_loader, val_loader
        """
        print("Preparing expert dataset...")
        
        # Create expert dataset
        dataset = create_expert_dataset(
            env_name=self.env_name,
            num_episodes=num_episodes,
            force_recollect=force_recollect
        )
        
        # Split into train and validation
        val_size = int(len(dataset) * self.training_config['validation_split'])
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (observations, actions) in enumerate(pbar):
            # Move to device
            observations = observations.to(self.device)
            actions = actions.to(self.device)
            
            # Handle different observation formats
            if observations.dtype == torch.uint8:
                observations = observations.float() / 255.0
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_actions = self.model(observations)
            loss = self.criterion(predicted_actions, actions)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item():.4f})
            
            # Log to tensorboard
            if batch_idx % 100 == 0:
                step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Training/BatchLoss', loss.item(), step)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for observations, actions in val_loader:
                # Move to device
                observations = observations.to(self.device)
                actions = actions.to(self.device)
                
                # Handle different observation formats
                if observations.dtype == torch.uint8:
                    observations = observations.float() / 255.0
                
                # Forward pass
                predicted_actions = self.model(observations)
                loss = self.criterion(predicted_actions, actions)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, 
              num_episodes: int = None,
              force_recollect: bool = False,
              save_best: bool = True) -> Dict[str, Any]:
        """
        Train the imitation learning model
        
        Args:
            num_episodes: Number of expert episodes to collect
            force_recollect: Whether to force recollection of expert data
            save_best: Whether to save the best model
            
        Returns:
            Training history dictionary
        """
        print("Starting Imitation Learning training...")
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(num_episodes, force_recollect)
        
        # Training loop
        for epoch in range(self.training_config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.training_config['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate_epoch(val_loader, epoch)
            self.val_losses.append(val_loss)
            
            # Log to tensorboard
            self.writer.add_scalar('Training/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Validation/EpochLoss', val_loss, epoch)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                if save_best:
                    self.save_model('best_model.pth')
                    print(f"New best model saved with val loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.training_config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Save checkpoint
            if (epoch + 1) % self.training_config['save_model_every'] == 0:
                self.save_model(f'checkpoint_epoch_{epoch + 1}.pth')
        
        # Final evaluation
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Plot training curves
        self.plot_training_curves()
        
        # Evaluate on environment
        self.evaluate_in_environment()
        
        training_history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'final_epoch': len(self.train_losses)
        }
        
        return training_history
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        models_dir = Config.PATHS['models_dir']
        os.makedirs(models_dir, exist_ok=True)
        
        filepath = os.path.join(models_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.model_config,
            'training_config': self.training_config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Model loaded from {filepath}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses, label='Validation Loss')
        plt.axhline(y=self.best_val_loss, color='r', linestyle='--', label=f'Best Val Loss: {self.best_val_loss:.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Progress')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plots_dir = os.path.join(Config.PATHS['logs_dir'], 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, 'imitation_learning_curves.png'))
        plt.show()
    
    def evaluate_in_environment(self, num_episodes: int = 5):
        """Evaluate the trained model in the environment"""
        print("\nEvaluating model in environment...")
        
        # Create evaluation environment
        env = create_environment(self.env_name, enable_domain_randomization=False)
        
        self.model.eval()
        episode_rewards = []
        episode_lengths = []
        
        with torch.no_grad():
            for episode in range(num_episodes):
                obs, info = env.reset()
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done:
                    # Prepare observation
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                    if obs_tensor.dtype == torch.uint8:
                        obs_tensor = obs_tensor.float() / 255.0
                    
                    # Get action from model
                    action = self.model(obs_tensor)
                    action = action.cpu().numpy().squeeze()
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward
                    episode_length += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        # Print statistics
        print(f"\nEvaluation Results (Average over {num_episodes} episodes):")
        print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        
        env.close()
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths)
        }


def train_imitation_model(env_name: str = 'highway',
                         num_episodes: int = 100,
                         force_recollect: bool = False) -> ImitationLearner:
    """
    Convenience function to train an imitation learning model
    
    Args:
        env_name: Environment name
        num_episodes: Number of expert episodes to collect
        force_recollect: Whether to force recollection of expert data
        
    Returns:
        Trained ImitationLearner
    """
    trainer = ImitationLearner(env_name)
    trainer.train(num_episodes=num_episodes, force_recollect=force_recollect)
    return trainer


if __name__ == "__main__":
    # Test imitation learning training
    trainer = train_imitation_model(
        env_name='highway',
        num_episodes=50,  # Small for testing
        force_recollect=True
    )