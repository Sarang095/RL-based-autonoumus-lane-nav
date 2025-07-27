#!/usr/bin/env python3
"""
Demo script for Webots Autonomous Driving Project
Demonstrates key components without requiring Webots installation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json


def demo_sensor_data_processing():
    """Demonstrate how sensor data is processed."""
    print("=" * 60)
    print("DEMO: Sensor Data Processing")
    print("=" * 60)
    
    # Simulate camera image (64x64x3)
    camera_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # Simulate distance sensor readings
    distance_sensors = {
        'front_sensor': 1.5,    # 1.5 meters ahead
        'rear_sensor': 0.8,     # 0.8 meters behind
        'left_sensor': 0.3,     # 0.3 meters to left
        'right_sensor': 2.0     # 2.0 meters to right
    }
    
    # Simulate orientation data (roll, pitch, yaw in radians)
    orientation = {
        'roll': 0.05,    # Slight roll
        'pitch': -0.02,  # Slight pitch down
        'yaw': 0.3       # Turned 0.3 radians to right
    }
    
    print("Simulated Sensor Data:")
    print(f"Camera image shape: {camera_image.shape}")
    print(f"Distance sensors: {distance_sensors}")
    print(f"Orientation (rad): {orientation}")
    
    # Process data as the agent would
    normalized_distances = {k: min(1.0, v / 2.0) for k, v in distance_sensors.items()}
    normalized_orientation = {k: v / np.pi for k, v in orientation.items()}
    
    print(f"\nNormalized distances: {normalized_distances}")
    print(f"Normalized orientation: {normalized_orientation}")
    
    return camera_image, distance_sensors, orientation


def demo_observation_space():
    """Demonstrate how observations are created for the RL agent."""
    print("\n" + "=" * 60)
    print("DEMO: Observation Space Construction")
    print("=" * 60)
    
    # Get simulated sensor data
    image, distances, orientation = demo_sensor_data_processing()
    
    # Build observation vector as in the real agent
    obs_list = []
    
    # Camera image (flattened and normalized)
    image_normalized = image.astype(np.float32) / 255.0
    obs_list.extend(image_normalized.flatten())
    print(f"Image data length: {len(image_normalized.flatten())}")
    
    # Distance sensors (normalized)
    for sensor_name in ['front_sensor', 'rear_sensor', 'left_sensor', 'right_sensor']:
        value = distances.get(sensor_name, 2.0)
        normalized_value = min(1.0, value / 2.0)
        obs_list.append(normalized_value)
    print(f"Distance sensors length: 4")
    
    # Orientation (normalized)
    for angle_name in ['roll', 'pitch', 'yaw']:
        angle = orientation.get(angle_name, 0.0)
        normalized_angle = angle / np.pi
        obs_list.append(normalized_angle)
    print(f"Orientation data length: 3")
    
    # Velocity estimates (simulated)
    current_throttle = 0.5
    current_steering = -0.2
    obs_list.extend([current_throttle, current_steering])
    print(f"Velocity data length: 2")
    
    observation = np.array(obs_list, dtype=np.float32)
    print(f"\nTotal observation vector length: {len(observation)}")
    print(f"Expected: {64*64*3 + 4 + 3 + 2} = {64*64*3 + 9}")
    
    return observation


def demo_reward_function():
    """Demonstrate the reward function used in RL training."""
    print("\n" + "=" * 60)
    print("DEMO: Reward Function")
    print("=" * 60)
    
    scenarios = [
        {"name": "Moving toward target", "distance_change": -0.5, "collision": False, "throttle": 0.7, "steering": 0.1},
        {"name": "Collision occurred", "distance_change": 0.0, "collision": True, "throttle": 0.5, "steering": 0.0},
        {"name": "Moving away from target", "distance_change": 0.3, "collision": False, "throttle": -0.2, "steering": 0.8},
        {"name": "Target reached", "distance_change": -2.0, "collision": False, "throttle": 0.3, "steering": 0.0},
    ]
    
    print("Scenario Analysis:")
    print("-" * 40)
    
    for scenario in scenarios:
        # Calculate reward components
        distance_reward = scenario["distance_change"] * 10  # +10 per meter of progress
        collision_penalty = -50 if scenario["collision"] else 0
        smooth_steering_bonus = -0.1 * abs(scenario["steering"])
        forward_motion_bonus = 0.5 * scenario["throttle"] if scenario["throttle"] > 0 else 0
        time_penalty = -0.01  # Small penalty per timestep
        
        total_reward = (distance_reward + collision_penalty + 
                       smooth_steering_bonus + forward_motion_bonus + time_penalty)
        
        print(f"\n{scenario['name']}:")
        print(f"  Distance progress: {distance_reward:+.1f}")
        print(f"  Collision penalty: {collision_penalty:+.1f}")
        print(f"  Smooth steering: {smooth_steering_bonus:+.2f}")
        print(f"  Forward motion: {forward_motion_bonus:+.2f}")
        print(f"  Time penalty: {time_penalty:+.2f}")
        print(f"  TOTAL REWARD: {total_reward:+.2f}")


def demo_training_data_format():
    """Demonstrate the format of collected training data."""
    print("\n" + "=" * 60)
    print("DEMO: Training Data Format")
    print("=" * 60)
    
    # Create sample training data
    timestamp = datetime.now().timestamp()
    sample_id = 42
    
    # Action data (what the human driver did)
    action_data = {
        'timestamp': timestamp,
        'sample_id': sample_id,
        'throttle': 0.6,
        'steering': -0.3,
        'image_filename': f"sample_{sample_id:06d}_{timestamp:.3f}.jpg"
    }
    
    # Sensor data (what the sensors measured)
    sensor_data = {
        'timestamp': timestamp,
        'sample_id': sample_id,
        'distances': {
            'front_sensor': 1.2,
            'rear_sensor': 1.8,
            'left_sensor': 0.5,
            'right_sensor': 1.1
        },
        'orientation': {
            'roll': 0.02,
            'pitch': -0.01,
            'yaw': 0.15
        },
        'control_state': {
            'throttle': 0.6,
            'steering': -0.3
        }
    }
    
    print("Sample Action Data (saved as JSON):")
    print(json.dumps(action_data, indent=2))
    
    print("\nSample Sensor Data (saved as JSON):")
    print(json.dumps(sensor_data, indent=2))
    
    print(f"\nCorresponding image saved as: {action_data['image_filename']}")
    print("Image contains: 128x128 RGB camera view")


def demo_network_architecture():
    """Demonstrate the neural network architecture used."""
    print("\n" + "=" * 60)
    print("DEMO: Neural Network Architecture")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn as nn
        
        # Simplified version of the PPO network
        class SimpleCarPolicy(nn.Module):
            def __init__(self, obs_dim=12297, action_dim=2):
                super().__init__()
                
                # Input layer
                self.fc1 = nn.Linear(obs_dim, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 64)
                
                # Output layer (steering and throttle)
                self.action_head = nn.Linear(64, action_dim)
                
                # Value function head
                self.value_head = nn.Linear(64, 1)
                
                self.relu = nn.ReLU()
                self.tanh = nn.Tanh()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.relu(self.fc3(x))
                
                actions = self.tanh(self.action_head(x))  # [-1, 1] range
                value = self.value_head(x)
                
                return actions, value
        
        # Create and display network
        obs_dim = 64*64*3 + 4 + 3 + 2  # Image + sensors + orientation + velocity
        net = SimpleCarPolicy(obs_dim)
        
        print(f"Network Input Dimension: {obs_dim}")
        print(f"Network Output Dimension: 2 (steering, throttle)")
        print("\nNetwork Architecture:")
        print(net)
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in net.parameters())
        print(f"\nTotal Parameters: {total_params:,}")
        
        # Demo forward pass
        dummy_input = torch.randn(1, obs_dim)
        with torch.no_grad():
            actions, value = net(dummy_input)
        
        print(f"\nExample prediction:")
        print(f"Steering: {actions[0, 0].item():.3f}")
        print(f"Throttle: {actions[0, 1].item():.3f}")
        print(f"State Value: {value[0, 0].item():.3f}")
        
    except ImportError:
        print("PyTorch not available - showing conceptual architecture:")
        print("Input Layer: 12,297 neurons (64x64x3 + 9)")
        print("Hidden Layer 1: 256 neurons (ReLU)")
        print("Hidden Layer 2: 128 neurons (ReLU)")
        print("Hidden Layer 3: 64 neurons (ReLU)")
        print("Action Output: 2 neurons (Tanh) -> [steering, throttle]")
        print("Value Output: 1 neuron (Linear) -> state value")


def demo_visualization():
    """Create some sample visualizations."""
    print("\n" + "=" * 60)
    print("DEMO: Training Progress Visualization")
    print("=" * 60)
    
    try:
        # Simulate training progress data
        episodes = np.arange(1, 101)
        
        # Simulated learning curve (improving over time with noise)
        base_reward = -20 * np.exp(-episodes / 30) + 50
        noise = np.random.normal(0, 10, len(episodes))
        rewards = base_reward + noise
        
        # Simulated episode lengths (should decrease as agent gets better)
        base_length = 800 * np.exp(-episodes / 40) + 200
        length_noise = np.random.normal(0, 50, len(episodes))
        episode_lengths = np.maximum(100, base_length + length_noise)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Reward plot
        ax1.plot(episodes, rewards, alpha=0.6, label='Episode Reward')
        ax1.plot(episodes, np.convolve(rewards, np.ones(10)/10, mode='same'), 
                'r-', linewidth=2, label='Moving Average (10 episodes)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Progress: Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Episode length plot  
        ax2.plot(episodes, episode_lengths, alpha=0.6, label='Episode Length')
        ax2.plot(episodes, np.convolve(episode_lengths, np.ones(10)/10, mode='same'),
                'g-', linewidth=2, label='Moving Average (10 episodes)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Training Progress: Episode Lengths')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = 'demo_training_progress.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Sample training visualization saved as: {output_file}")
        print("This shows how reward increases and episode length decreases during training.")
        
    except ImportError:
        print("Matplotlib not available - skipping visualization demo")


def main():
    """Run all demonstrations."""
    print("🚗 WEBOTS AUTONOMOUS DRIVING - DEMO")
    print("This demo shows key components without requiring Webots")
    print("=" * 60)
    
    # Run all demos
    demos = [
        demo_sensor_data_processing,
        demo_observation_space,
        demo_reward_function,
        demo_training_data_format,
        demo_network_architecture,
        demo_visualization
    ]
    
    for demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"Demo error: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETED!")
    print("=" * 60)
    print("\nTo run the actual simulation:")
    print("1. Install Webots from https://cyberbotics.com/")
    print("2. Run: python setup_project.py")
    print("3. Run: python launch_simulation.py")
    print("\nFor more information, see QUICKSTART.md")


if __name__ == "__main__":
    main()