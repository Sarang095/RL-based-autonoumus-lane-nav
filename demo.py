#!/usr/bin/env python3
"""
Quick Demonstration of Vision-Based Autonomous Driving Agent
Shows the key capabilities of the system with minimal setup.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from src.environments.vision_wrapper import HighwayVisionEnv
from src.environments.domain_randomizer import DomainRandomizer
from src.models.cnn_policy import CNNPolicy


def demo_vision_processing():
    """Demonstrate the vision processing pipeline."""
    print("🎥 Vision Processing Demo")
    print("-" * 30)
    
    # Create vision environment
    env = HighwayVisionEnv.create_env(
        env_name="highway-v0",
        observation_type="rgb",
        image_shape=(84, 84),
        stack_frames=4
    )
    
    print(f"✅ Environment: {env.observation_space}")
    
    # Get sample observation
    obs, info = env.reset()
    print(f"✅ Observation shape: {obs.shape}")
    print(f"✅ Value range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Take a few steps
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Step {i+1}: Action={action}, Reward={reward:.3f}")
    
    env.close()
    print("✅ Vision processing demo completed!\n")


def demo_cnn_architecture():
    """Demonstrate the CNN architecture."""
    print("🧠 CNN Architecture Demo")
    print("-" * 30)
    
    # Create CNN policy
    policy = CNNPolicy(
        input_channels=12,
        input_height=84,
        input_width=84,
        action_dim=5,
        use_attention=True
    )
    
    print(f"✅ Model parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 12, 84, 84)
    
    with torch.no_grad():
        output = policy(dummy_input, return_aux=True)
    
    print(f"✅ Forward pass successful:")
    print(f"   Action probs: {output['action_probs'].shape}")
    print(f"   Value: {output['value'].shape}")
    print(f"   Features: {output['features'].shape}")
    print(f"   Auxiliary outputs: {len([k for k in output.keys() if 'pred' in k])}")
    
    # Sample actions
    actions, log_probs = policy.get_action(dummy_input)
    print(f"✅ Action sampling: {actions.tolist()}")
    print("✅ CNN architecture demo completed!\n")


def demo_domain_randomization():
    """Demonstrate domain randomization."""
    print("🎲 Domain Randomization Demo")
    print("-" * 30)
    
    # Create base environment
    base_env = HighwayVisionEnv.create_env("highway-v0")
    
    # Add domain randomization
    env = DomainRandomizer(
        base_env,
        randomize_traffic=True,
        randomize_weather=True,
        randomize_behavior=True
    )
    
    print("✅ Domain randomizer created")
    
    # Show randomization across resets
    for i in range(3):
        obs, info = env.reset()
        randomization = info.get("domain_randomization", {})
        print(f"   Reset {i+1}:")
        print(f"     Traffic: {randomization.get('vehicles_count', 'N/A')}")
        print(f"     Weather: {randomization.get('weather', 'N/A')}")
        
        # Take a few steps to see environment in action
        for _ in range(2):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    
    env.close()
    print("✅ Domain randomization demo completed!\n")


def demo_training_pipeline():
    """Demonstrate the training pipeline setup."""
    print("🏋️ Training Pipeline Demo")
    print("-" * 30)
    
    # Show what the training pipeline would do
    print("Training pipeline includes:")
    print("1. 🚗 Expert demonstration collection")
    print("2. 🧠 Imitation learning with CNN")
    print("3. 🚀 PPO reinforcement learning")
    print("4. 📊 Multi-agent evaluation")
    
    print("\nTo run the full training pipeline:")
    print("   python train_vision_agent.py --phase all")
    
    print("\nOr run individual phases:")
    print("   python train_vision_agent.py --phase collect")
    print("   python train_vision_agent.py --phase imitation")
    print("   python train_vision_agent.py --phase ppo")
    print("   python train_vision_agent.py --phase evaluate")
    
    print("✅ Training pipeline demo completed!\n")


def main():
    """Run all demonstrations."""
    print("🚀 Vision-Based Autonomous Driving Agent Demo")
    print("=" * 50)
    print("This demo showcases the key components of our system:\n")
    
    try:
        demo_vision_processing()
        demo_cnn_architecture()
        demo_domain_randomization()
        demo_training_pipeline()
        
        print("=" * 50)
        print("✅ All demos completed successfully!")
        print("\n🎯 Key Features Demonstrated:")
        print("   • Vision-based observation processing")
        print("   • CNN architecture with attention")
        print("   • Domain randomization capabilities")
        print("   • Complete training pipeline")
        
        print("\n📚 Next Steps:")
        print("   • Run full tests: python test_vision_env.py")
        print("   • Start training: python train_vision_agent.py")
        print("   • Check documentation in README.md")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()