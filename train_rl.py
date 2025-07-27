"""
Reinforcement Learning Training Script for Autonomous Driving
Uses Stable-Baselines3 PPO with a custom Webots gym environment.
"""

import gym
from gym import spaces
import numpy as np
import cv2
import os
import time
import json
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt


class WebotsCarEnv(gym.Env):
    """
    Custom OpenAI Gym environment for Webots autonomous driving simulation.
    Integrates with Webots supervisor and agent controllers.
    """
    
    def __init__(self, 
                 webots_world_path=None,
                 max_episode_steps=1000,
                 target_scenarios=['parking', 'intersection', 'roundabout']):
        """
        Initialize the Webots car environment.
        
        Args:
            webots_world_path (str): Path to the Webots world file
            max_episode_steps (int): Maximum steps per episode
            target_scenarios (list): List of scenarios to train on
        """
        super(WebotsCarEnv, self).__init__()
        
        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.target_scenarios = target_scenarios
        self.current_scenario = 'parking'  # Default scenario
        
        # Action space: [steering_angle, throttle]
        # steering_angle: -1 (full left) to 1 (full right)
        # throttle: -1 (full reverse) to 1 (full forward)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: 
        # - Camera image: 64x64x3 (downsampled RGB)
        # - Distance sensors: 4 values (front, rear, left, right)
        # - Orientation: 3 values (roll, pitch, yaw)
        # - Velocity info: 2 values (linear, angular)
        
        image_shape = (64, 64, 3)
        sensor_dim = 4  # distance sensors
        orientation_dim = 3  # roll, pitch, yaw
        velocity_dim = 2  # linear and angular velocity estimates
        
        # Total observation dimension
        obs_dim = np.prod(image_shape) + sensor_dim + orientation_dim + velocity_dim
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Webots connection setup
        self.webots_supervisor = None
        self.webots_agent = None
        self._setup_webots_connection()
        
        # Scenario management
        self.scenario_targets = {
            'parking': {'target_pos': [10.0, 0.0, 5.0], 'tolerance': 2.0},
            'intersection': {'target_pos': [0.0, 0.0, 20.0], 'tolerance': 3.0},
            'roundabout': {'target_pos': [-10.0, 0.0, 15.0], 'tolerance': 3.0}
        }
        
        # Reward calculation parameters
        self.previous_distance_to_target = None
        self.previous_position = None
        self.collision_threshold = 0.5  # Distance sensor threshold for collision
        
        # Episode statistics
        self.episode_rewards = []
        self.episode_lengths = []
        
        print("WebotsCarEnv initialized successfully")
    
    def _setup_webots_connection(self):
        """Setup connection to Webots supervisor and agent controllers."""
        try:
            # In a real implementation, you would establish communication
            # with the Webots controllers through sockets, shared memory, or files
            
            # Placeholder for Webots connection
            print("Setting up Webots connection...")
            
            # This would typically involve:
            # 1. Starting Webots simulation
            # 2. Connecting to supervisor controller
            # 3. Connecting to agent controller
            # 4. Setting up communication channels
            
            self.webots_connected = True
            print("Webots connection established")
            
        except Exception as e:
            print(f"Failed to connect to Webots: {e}")
            self.webots_connected = False
    
    def _get_obs(self):
        """
        Get current observation from Webots simulation.
        
        Returns:
            np.ndarray: Flattened observation vector
        """
        if not self.webots_connected:
            # Return dummy observation for testing
            obs_dim = self.observation_space.shape[0]
            return np.random.random(obs_dim).astype(np.float32)
        
        try:
            # Get sensor data from Webots agent
            sensor_data = self._get_webots_sensor_data()
            
            # Process camera image
            camera_image = sensor_data.get('camera_image')
            if camera_image is not None:
                # Resize to 64x64 and normalize
                processed_image = cv2.resize(camera_image, (64, 64))
                processed_image = processed_image.astype(np.float32) / 255.0
                image_flat = processed_image.flatten()
            else:
                image_flat = np.zeros(64 * 64 * 3, dtype=np.float32)
            
            # Process distance sensors
            distances = sensor_data.get('distances', {})
            distance_values = [
                distances.get('front_sensor', 1.0),
                distances.get('rear_sensor', 1.0),
                distances.get('left_sensor', 1.0),
                distances.get('right_sensor', 1.0)
            ]
            distance_array = np.array(distance_values, dtype=np.float32)
            
            # Process orientation
            orientation = sensor_data.get('orientation', {})
            orientation_values = [
                orientation.get('roll', 0.0),
                orientation.get('pitch', 0.0),
                orientation.get('yaw', 0.0)
            ]
            orientation_array = np.array(orientation_values, dtype=np.float32)
            
            # Estimate velocity (simplified)
            if self.previous_position is not None:
                current_pos = self._get_agent_position()
                velocity = np.linalg.norm(np.array(current_pos) - np.array(self.previous_position))
                angular_velocity = abs(orientation_values[2] - getattr(self, 'previous_yaw', 0.0))
            else:
                velocity = 0.0
                angular_velocity = 0.0
            
            velocity_array = np.array([velocity, angular_velocity], dtype=np.float32)
            
            # Combine all observations
            observation = np.concatenate([
                image_flat,
                distance_array,
                orientation_array,
                velocity_array
            ])
            
            return observation.astype(np.float32)
            
        except Exception as e:
            print(f"Error getting observation: {e}")
            # Return zero observation on error
            obs_dim = self.observation_space.shape[0]
            return np.zeros(obs_dim, dtype=np.float32)
    
    def _get_webots_sensor_data(self):
        """
        Get sensor data from Webots agent.
        
        Returns:
            dict: Dictionary containing sensor readings
        """
        # Placeholder implementation
        # In practice, this would communicate with the agent_driver.py controller
        
        dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        sensor_data = {
            'camera_image': dummy_image,
            'distances': {
                'front_sensor': np.random.random(),
                'rear_sensor': np.random.random(),
                'left_sensor': np.random.random(),
                'right_sensor': np.random.random()
            },
            'orientation': {
                'roll': np.random.random() * 0.1,
                'pitch': np.random.random() * 0.1,
                'yaw': np.random.random() * np.pi * 2
            }
        }
        
        return sensor_data
    
    def _get_agent_position(self):
        """
        Get current agent position from Webots.
        
        Returns:
            list: [x, y, z] coordinates
        """
        # Placeholder implementation
        # In practice, this would get position from supervisor controller
        return [np.random.random() * 20 - 10, 0.0, np.random.random() * 20]
    
    def _send_action_to_webots(self, action):
        """
        Send action to Webots agent controller.
        
        Args:
            action (np.ndarray): [steering_angle, throttle]
        """
        if not self.webots_connected:
            return
        
        # Placeholder implementation
        # In practice, this would send action to agent_driver.py
        steering_angle, throttle = action
        print(f"Sending action: steering={steering_angle:.3f}, throttle={throttle:.3f}")
    
    def _calculate_reward(self, action, sensor_data):
        """
        Calculate reward based on current state and action.
        
        Args:
            action (np.ndarray): Current action
            sensor_data (dict): Current sensor readings
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        # Get current position
        current_pos = self._get_agent_position()
        
        # Target position for current scenario
        target_pos = self.scenario_targets[self.current_scenario]['target_pos']
        tolerance = self.scenario_targets[self.current_scenario]['tolerance']
        
        # Distance to target reward
        distance_to_target = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        
        if self.previous_distance_to_target is not None:
            # Reward for getting closer to target
            distance_improvement = self.previous_distance_to_target - distance_to_target
            reward += distance_improvement * 10.0
        
        self.previous_distance_to_target = distance_to_target
        
        # Success reward
        if distance_to_target < tolerance:
            reward += 100.0  # Large reward for reaching target
            print(f"Target reached! Distance: {distance_to_target:.2f}")
        
        # Collision penalty
        distances = sensor_data.get('distances', {})
        min_distance = min(distances.values()) if distances else 1.0
        
        if min_distance < self.collision_threshold:
            reward -= 50.0  # Penalty for collision
            print(f"Collision detected! Min distance: {min_distance:.2f}")
        
        # Smooth driving reward
        steering_angle, throttle = action
        
        # Penalty for aggressive steering
        reward -= abs(steering_angle) * 0.1
        
        # Small positive reward for forward motion
        if throttle > 0:
            reward += throttle * 0.5
        
        # Penalty for staying still
        if abs(throttle) < 0.01:
            reward -= 0.1
        
        # Time penalty to encourage efficiency
        reward -= 0.01
        
        return reward
    
    def _is_episode_done(self, sensor_data):
        """
        Check if episode should terminate.
        
        Args:
            sensor_data (dict): Current sensor readings
            
        Returns:
            bool: True if episode is done
        """
        # Maximum steps reached
        if self.current_step >= self.max_episode_steps:
            return True
        
        # Target reached
        current_pos = self._get_agent_position()
        target_pos = self.scenario_targets[self.current_scenario]['target_pos']
        tolerance = self.scenario_targets[self.current_scenario]['tolerance']
        distance_to_target = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        
        if distance_to_target < tolerance:
            return True
        
        # Collision detected
        distances = sensor_data.get('distances', {})
        min_distance = min(distances.values()) if distances else 1.0
        
        if min_distance < self.collision_threshold:
            return True
        
        return False
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (np.ndarray): Action to execute
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        self.current_step += 1
        
        # Send action to Webots
        self._send_action_to_webots(action)
        
        # Step the simulation (in practice, this would wait for Webots step)
        time.sleep(0.01)  # Simulate timestep
        
        # Get new observation
        observation = self._get_obs()
        
        # Get sensor data for reward calculation
        sensor_data = self._get_webots_sensor_data()
        
        # Calculate reward
        reward = self._calculate_reward(action, sensor_data)
        
        # Check if episode is done
        done = self._is_episode_done(sensor_data)
        
        # Additional info
        info = {
            'episode_step': self.current_step,
            'scenario': self.current_scenario,
            'distance_to_target': self.previous_distance_to_target or 0.0
        }
        
        return observation, reward, done, info
    
    def reset(self):
        """
        Reset the environment for a new episode.
        
        Returns:
            np.ndarray: Initial observation
        """
        # Reset episode counters
        self.current_step = 0
        self.previous_distance_to_target = None
        self.previous_position = None
        
        # Randomly select scenario
        self.current_scenario = np.random.choice(self.target_scenarios)
        print(f"Starting new episode with scenario: {self.current_scenario}")
        
        # Reset Webots simulation
        self._reset_webots_simulation()
        
        # Get initial observation
        observation = self._get_obs()
        
        return observation
    
    def _reset_webots_simulation(self):
        """Reset the Webots simulation to initial state."""
        if not self.webots_connected:
            return
        
        # In practice, this would send reset command to supervisor_controller.py
        print("Resetting Webots simulation...")
        
        # Set traffic lights based on scenario
        if self.current_scenario == 'intersection':
            # Set appropriate traffic light configuration for intersection
            pass
        elif self.current_scenario == 'roundabout':
            # Configure roundabout scenario
            pass
        elif self.current_scenario == 'parking':
            # Configure parking scenario
            pass
    
    def render(self, mode='human'):
        """Render the environment (optional)."""
        if mode == 'human':
            # In practice, Webots provides its own visualization
            pass
        return None
    
    def close(self):
        """Clean up resources."""
        if self.webots_connected:
            print("Closing Webots connection...")
            # Close communication channels


class TrainingCallback(BaseCallback):
    """Custom callback for monitoring training progress."""
    
    def __init__(self, log_dir, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log episode statistics
        if 'episode' in self.locals.get('infos', [{}])[0]:
            episode_info = self.locals['infos'][0]['episode']
            self.episode_rewards.append(episode_info['r'])
            self.episode_lengths.append(episode_info['l'])
            
            if len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episodes: {len(self.episode_rewards)}, Avg Reward (last 10): {avg_reward:.2f}")
        
        return True


def main():
    """Main training function."""
    print("Starting Webots RL Training...")
    
    # Create log directory
    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    env = WebotsCarEnv(
        max_episode_steps=1000,
        target_scenarios=['parking', 'intersection', 'roundabout']
    )
    
    # Check environment
    print("Checking environment...")
    check_env(env)
    print("Environment check passed!")
    
    # Wrap environment
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    
    # Create PPO model
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device='auto'
    )
    
    # Setup callbacks
    callback = TrainingCallback(log_dir)
    
    # Training parameters
    total_timesteps = 100000  # Adjust based on your needs
    save_freq = 10000
    
    print(f"Starting training for {total_timesteps} timesteps...")
    
    try:
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name="ppo_webots_car"
        )
        
        # Save the final model
        model_path = os.path.join(log_dir, "final_model")
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Test the trained model
        print("Testing trained model...")
        test_episodes = 5
        
        obs = env.reset()
        for episode in range(test_episodes):
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}")
            obs = env.reset()
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    except Exception as e:
        print(f"Training error: {e}")
    
    finally:
        # Clean up
        env.close()
        print("Training completed!")


if __name__ == "__main__":
    main()