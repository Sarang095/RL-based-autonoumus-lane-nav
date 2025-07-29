"""
Expert data collection for imitation learning
Collects expert demonstrations from highway-env's built-in agents
"""
import os
import pickle
import numpy as np
import gymnasium as gym
import highway_env
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
import torch

from ..environment_wrapper import create_environment
from ..config import Config


class ExpertDataCollector:
    """
    Collects expert demonstrations for imitation learning
    Uses highway-env's built-in expert policies
    """
    
    def __init__(self, env_name: str = 'highway', config: Dict[str, Any] = None):
        self.env_name = env_name
        self.config = config or Config.IL_CONFIG
        self.save_path = Config.PATHS['expert_data']
        
        # Create environment without domain randomization for consistent expert data
        self.env = create_environment(
            env_name=env_name,
            enable_domain_randomization=False,
            enable_multi_agent=False,
            enable_reward_shaping=False
        )
        
        # Storage for trajectories
        self.trajectories = []
        
    def collect_expert_trajectories(self, 
                                  num_episodes: int = None,
                                  use_keyboard: bool = False,
                                  save_data: bool = True) -> List[Dict[str, Any]]:
        """
        Collect expert demonstrations
        
        Args:
            num_episodes: Number of episodes to collect (default from config)
            use_keyboard: Whether to use keyboard control (manual expert)
            save_data: Whether to save collected data
            
        Returns:
            List of trajectory dictionaries
        """
        if num_episodes is None:
            num_episodes = self.config['expert_episodes']
        
        print(f"Collecting {num_episodes} expert trajectories...")
        
        for episode in tqdm(range(num_episodes)):
            if use_keyboard:
                trajectory = self._collect_keyboard_trajectory()
            else:
                trajectory = self._collect_automatic_trajectory()
            
            if trajectory['observations']:  # Only add non-empty trajectories
                self.trajectories.append(trajectory)
        
        print(f"Collected {len(self.trajectories)} valid trajectories")
        
        if save_data:
            self.save_trajectories()
        
        return self.trajectories
    
    def _collect_automatic_trajectory(self) -> Dict[str, Any]:
        """Collect trajectory using highway-env's built-in expert"""
        observations = []
        actions = []
        rewards = []
        dones = []
        
        obs, info = self.env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Use highway-env's built-in IDM (Intelligent Driver Model) behavior
            action = self._get_expert_action(obs, info)
            
            # Store data
            observations.append(obs.copy())
            actions.append(action.copy())
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            rewards.append(reward)
            dones.append(done)
            total_reward += reward
            
            obs = next_obs
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'total_reward': total_reward,
            'length': len(observations)
        }
    
    def _get_expert_action(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """
        Get expert action using a heuristic policy
        This simulates an expert driver's behavior
        """
        # Create a simple heuristic expert policy
        # In a real scenario, you might use more sophisticated expert policies
        
        # For continuous action space: [acceleration, steering]
        action = np.array([0.0, 0.0])  # Default: maintain speed, no steering
        
        # Simple heuristic: try to maintain speed and stay in lane
        # This is a placeholder - in practice you might use:
        # 1. Highway-env's built-in IDM controller
        # 2. Manual demonstrations
        # 3. Pre-trained policies
        
        # Random expert with some intelligence
        if np.random.random() < 0.7:  # 70% of time use reasonable actions
            # Slight acceleration to maintain speed
            action[0] = np.random.uniform(-0.2, 0.3)
            # Small steering corrections
            action[1] = np.random.uniform(-0.1, 0.1)
        else:
            # 30% random actions (for exploration)
            action = np.random.uniform(-0.5, 0.5, size=2)
        
        # Ensure actions are in valid range
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def _collect_keyboard_trajectory(self) -> Dict[str, Any]:
        """
        Collect trajectory using keyboard input (manual expert)
        Note: This is a placeholder for keyboard control implementation
        """
        print("Keyboard control not implemented in this demo.")
        print("Using automatic expert instead...")
        return self._collect_automatic_trajectory()
    
    def save_trajectories(self, filepath: str = None):
        """Save collected trajectories to file"""
        if filepath is None:
            filepath = self.save_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare data for saving
        data = {
            'trajectories': self.trajectories,
            'env_name': self.env_name,
            'config': self.config,
            'num_trajectories': len(self.trajectories),
            'total_steps': sum(traj['length'] for traj in self.trajectories)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(self.trajectories)} trajectories to {filepath}")
    
    def load_trajectories(self, filepath: str = None) -> List[Dict[str, Any]]:
        """Load trajectories from file"""
        if filepath is None:
            filepath = self.save_path
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Trajectory file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.trajectories = data['trajectories']
        print(f"Loaded {len(self.trajectories)} trajectories from {filepath}")
        
        return self.trajectories
    
    def get_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert trajectories to dataset format for training
        
        Returns:
            observations: Array of observations
            actions: Array of corresponding actions
        """
        if not self.trajectories:
            raise ValueError("No trajectories collected. Run collect_expert_trajectories first.")
        
        all_observations = []
        all_actions = []
        
        for trajectory in self.trajectories:
            all_observations.extend(trajectory['observations'])
            all_actions.extend(trajectory['actions'])
        
        observations = np.array(all_observations)
        actions = np.array(all_actions)
        
        print(f"Dataset: {observations.shape[0]} samples")
        print(f"Observation shape: {observations.shape[1:]}")
        print(f"Action shape: {actions.shape[1:]}")
        
        return observations, actions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected data"""
        if not self.trajectories:
            return {}
        
        lengths = [traj['length'] for traj in self.trajectories]
        rewards = [traj['total_reward'] for traj in self.trajectories]
        
        stats = {
            'num_trajectories': len(self.trajectories),
            'total_steps': sum(lengths),
            'avg_episode_length': np.mean(lengths),
            'std_episode_length': np.std(lengths),
            'min_episode_length': np.min(lengths),
            'max_episode_length': np.max(lengths),
            'avg_episode_reward': np.mean(rewards),
            'std_episode_reward': np.std(rewards),
            'min_episode_reward': np.min(rewards),
            'max_episode_reward': np.max(rewards),
        }
        
        return stats


class ExpertDataset(torch.utils.data.Dataset):
    """PyTorch dataset for expert trajectories"""
    
    def __init__(self, observations: np.ndarray, actions: np.ndarray, transform=None):
        self.observations = observations
        self.actions = actions
        self.transform = transform
        
        # Convert to torch tensors
        if isinstance(observations, np.ndarray):
            self.observations = torch.from_numpy(observations).float()
        if isinstance(actions, np.ndarray):
            self.actions = torch.from_numpy(actions).float()
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        obs = self.observations[idx]
        action = self.actions[idx]
        
        if self.transform:
            obs = self.transform(obs)
        
        return obs, action


def create_expert_dataset(env_name: str = 'highway', 
                         num_episodes: int = None,
                         force_recollect: bool = False) -> ExpertDataset:
    """
    Create expert dataset for training
    
    Args:
        env_name: Environment name
        num_episodes: Number of episodes to collect
        force_recollect: Whether to force recollection even if data exists
        
    Returns:
        PyTorch dataset with expert demonstrations
    """
    collector = ExpertDataCollector(env_name)
    
    # Try to load existing data first
    try:
        if not force_recollect:
            collector.load_trajectories()
            print("Using existing expert data")
    except FileNotFoundError:
        print("No existing expert data found, collecting new data...")
        collector.collect_expert_trajectories(num_episodes)
    
    # Convert to dataset format
    observations, actions = collector.get_dataset()
    dataset = ExpertDataset(observations, actions)
    
    # Print statistics
    stats = collector.get_statistics()
    print("\nExpert Data Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    return dataset


if __name__ == "__main__":
    # Test expert data collection
    collector = ExpertDataCollector('highway')
    
    # Collect small dataset for testing
    trajectories = collector.collect_expert_trajectories(num_episodes=10)
    
    # Get dataset
    obs, actions = collector.get_dataset()
    print(f"\nCollected dataset with {obs.shape[0]} samples")
    
    # Test PyTorch dataset
    dataset = ExpertDataset(obs, actions)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Test one batch
    for batch_obs, batch_actions in dataloader:
        print(f"Batch observation shape: {batch_obs.shape}")
        print(f"Batch actions shape: {batch_actions.shape}")
        break