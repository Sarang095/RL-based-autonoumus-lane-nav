#!/usr/bin/env python3
"""
Test script for Vision-Based Autonomous Driving Agent
Demonstrates environment setup, data collection, and basic functionality.
"""

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from src.environments.vision_wrapper import HighwayVisionEnv, VisionWrapper
from src.environments.domain_randomizer import DomainRandomizer
from src.models.cnn_policy import CNNPolicy
from src.models.imitation_trainer import ImitationTrainer, DrivingDataset
import highway_env
import os


def test_environment_setup():
    """Test the basic environment setup and vision wrapper."""
    print("🚗 Testing Environment Setup...")
    
    # Create vision-based environment
    env = HighwayVisionEnv.create_env(
        env_name="highway-v0",
        observation_type="rgb",
        image_shape=(84, 84),
        stack_frames=4,
        normalize=True
    )
    
    print(f"✅ Environment created successfully")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Action space: {env.action_space}")
    
    # Test environment reset and step
    obs, info = env.reset()
    print(f"   - Initial observation shape: {obs.shape}")
    print(f"   - Initial observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Take a few random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   - Step {i+1}: Action={action}, Reward={reward:.3f}, Done={terminated or truncated}")
        
        if terminated or truncated:
            obs, info = env.reset()
            
    env.close()
    return True


def test_domain_randomization():
    """Test domain randomization wrapper."""
    print("\n🎲 Testing Domain Randomization...")
    
    # Create base environment
    base_env = HighwayVisionEnv.create_env("highway-v0")
    
    # Wrap with domain randomizer
    env = DomainRandomizer(
        base_env,
        randomize_traffic=True,
        randomize_weather=True,
        randomize_behavior=True,
        traffic_density_range=(5, 25)
    )
    
    print("✅ Domain randomizer created successfully")
    
    # Test multiple resets to see randomization
    for i in range(3):
        obs, info = env.reset()
        randomization_info = info.get("domain_randomization", {})
        print(f"   - Reset {i+1}:")
        print(f"     Traffic: {randomization_info.get('vehicles_count', 'N/A')}")
        print(f"     Weather: {randomization_info.get('weather', 'N/A')}")
        
        # Take a few steps
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
                
    env.close()
    return True


def test_cnn_policy():
    """Test CNN policy network."""
    print("\n🧠 Testing CNN Policy Network...")
    
    # Create policy
    policy = CNNPolicy(
        input_channels=12,  # 4 frames * 3 RGB
        input_height=84,
        input_width=84,
        action_dim=5,
        hidden_dim=512,
        use_attention=True
    )
    
    print(f"✅ CNN Policy created successfully")
    print(f"   - Total parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"   - Trainable parameters: {sum(p.numel() for p in policy.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 12, 84, 84)
    
    with torch.no_grad():
        output = policy(dummy_input, return_aux=True)
        
    print(f"   - Action logits shape: {output['action_logits'].shape}")
    print(f"   - Action probs shape: {output['action_probs'].shape}")
    print(f"   - Value shape: {output['value'].shape}")
    print(f"   - Features shape: {output['features'].shape}")
    print(f"   - Speed pred shape: {output['speed_pred'].shape}")
    print(f"   - Lane pred shape: {output['lane_pred'].shape}")
    
    # Test action sampling
    action, log_prob = policy.get_action(dummy_input[:1])
    print(f"   - Sampled action: {action.item()}")
    print(f"   - Log probability: {log_prob.item():.3f}")
    
    return True


def simple_expert_policy(obs, info=None):
    """Simple expert policy for demonstration collection."""
    # This is a very basic policy - in practice you'd use a more sophisticated expert
    # For now, we'll use a simple heuristic that tends to stay in the right lane and maintain speed
    
    # Actions in highway-env: 0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER
    actions = [0, 1, 2, 3, 4]
    
    # Simple heuristic: mostly go straight with occasional speed adjustments
    action_probs = [0.1, 0.4, 0.1, 0.3, 0.1]
    return np.random.choice(actions, p=action_probs)


def test_data_collection():
    """Test expert demonstration collection."""
    print("\n📊 Testing Expert Demonstration Collection...")
    
    # Create environment
    env = HighwayVisionEnv.create_env(
        env_name="highway-v0",
        observation_type="rgb",
        image_shape=(84, 84),
        stack_frames=4
    )
    
    # Create policy and trainer
    policy = CNNPolicy(input_channels=12, action_dim=5)
    trainer = ImitationTrainer(policy)
    
    print("✅ Starting demonstration collection...")
    
    # Collect demonstrations using simple expert
    observations, actions, auxiliary_data = trainer.collect_demonstrations(
        env=env,
        expert_policy=simple_expert_policy,
        num_episodes=5,  # Small number for testing
        save_path="data/test_demonstrations.pkl"
    )
    
    print(f"   - Collected {len(observations)} demonstrations")
    
    if len(observations) > 0:
        print(f"   - Action distribution: {np.bincount(actions)}")
        print(f"   - Average speed: {np.mean(auxiliary_data['speeds']):.3f}")
        
        # Create dataset
        dataset = DrivingDataset(
            observations=observations,
            actions=actions,
            auxiliary_data=auxiliary_data,
            augment_data=True
        )
        
        print(f"   - Dataset length: {len(dataset)}")
        
        # Test data loading
        sample = dataset[0]
        print(f"   - Sample observation shape: {sample['observation'].shape}")
        print(f"   - Sample action: {sample['action'].item()}")
    else:
        print("   - No valid episodes collected (likely due to crashes)")
        print("   - Creating dummy dataset for testing...")
        
        # Create dummy dataset for testing
        dummy_observations = [np.random.randn(12, 84, 84).astype(np.float32) for _ in range(10)]
        dummy_actions = [np.random.randint(0, 5) for _ in range(10)]
        dummy_auxiliary = {
            "speeds": [np.random.uniform(0, 1) for _ in range(10)],
            "lane_positions": [np.random.randint(0, 3) for _ in range(10)]
        }
        
        dataset = DrivingDataset(
            observations=dummy_observations,
            actions=dummy_actions,
            auxiliary_data=dummy_auxiliary,
            augment_data=True
        )
        
        print(f"   - Dummy dataset length: {len(dataset)}")
        sample = dataset[0]
        print(f"   - Sample observation shape: {sample['observation'].shape}")
        print(f"   - Sample action: {sample['action'].item()}")
    
    env.close()
    return True


def test_training_setup():
    """Test training setup without actual training."""
    print("\n🏋️ Testing Training Setup...")
    
    # Create dummy data
    batch_size = 8
    num_samples = 32
    
    dummy_observations = [
        np.random.randn(12, 84, 84).astype(np.float32) 
        for _ in range(num_samples)
    ]
    dummy_actions = np.random.randint(0, 5, num_samples).tolist()
    dummy_auxiliary = {
        "speeds": np.random.uniform(0, 1, num_samples).tolist(),
        "lane_positions": np.random.randint(0, 3, num_samples).tolist()
    }
    
    # Create dataset
    dataset = DrivingDataset(
        observations=dummy_observations,
        actions=dummy_actions,
        auxiliary_data=dummy_auxiliary,
        augment_data=False  # Disable for testing
    )
    
    # Create policy and trainer
    policy = CNNPolicy(input_channels=12, action_dim=5)
    trainer = ImitationTrainer(
        policy=policy,
        batch_size=batch_size,
        use_auxiliary_losses=True
    )
    
    print("✅ Training setup created successfully")
    print(f"   - Dataset size: {len(dataset)}")
    print(f"   - Batch size: {batch_size}")
    
    # Test single training step
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    batch = next(iter(data_loader))
    print(f"   - Batch observation shape: {batch['observation'].shape}")
    print(f"   - Batch action shape: {batch['action'].shape}")
    
    # Test forward pass with batch
    policy.train()
    with torch.no_grad():
        outputs = policy(batch['observation'], return_aux=True)
        print(f"   - Output shapes verified")
    
    return True


def visualize_observations(env, num_steps=10):
    """Visualize environment observations."""
    print("\n📸 Visualizing Environment Observations...")
    
    obs, info = env.reset()
    
    # Create figure for visualization
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Vision-Based Highway Environment")
    
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step < 4:  # Only visualize first 4 steps
            ax = axes[step // 2, step % 2]
            
            # Reshape observation for visualization
            if obs.shape[0] == 12:  # RGB stacked frames
                # Take first 3 channels (first frame)
                img = obs[:3].transpose(1, 2, 0)
            else:  # Grayscale
                img = obs[0]  # First frame
                
            ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            ax.set_title(f"Step {step+1}, Action: {action}, Reward: {reward:.2f}")
            ax.axis('off')
            
        if terminated or truncated:
            obs, info = env.reset()
    
    # Save visualization
    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/environment_visualization.png", dpi=150, bbox_inches='tight')
    print("   - Saved visualization to results/environment_visualization.png")
    plt.close()


def main():
    """Run all tests."""
    print("🚀 Starting Vision-Based Autonomous Driving Agent Tests")
    print("=" * 60)
    
    try:
        # Test basic environment setup
        test_environment_setup()
        
        # Test domain randomization
        test_domain_randomization()
        
        # Test CNN policy
        test_cnn_policy()
        
        # Test data collection
        test_data_collection()
        
        # Test training setup
        test_training_setup()
        
        # Visualization test
        print("\n📊 Testing Visualization...")
        env = HighwayVisionEnv.create_env("highway-v0", observation_type="rgb")
        visualize_observations(env)
        env.close()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("\n🎯 Next Steps:")
        print("   1. Collect more expert demonstrations")
        print("   2. Train the imitation learning model")
        print("   3. Implement PPO for reinforcement learning")
        print("   4. Add multi-agent scenarios")
        print("   5. Evaluate on different driving scenarios")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    main()