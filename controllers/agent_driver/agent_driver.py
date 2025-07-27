"""
Agent Driver Controller for Autonomous Driving
Handles both manual data collection (IL) and AI-driven inference modes.
"""

from controller import Robot, Camera, RotationalMotor, DistanceSensor, InertialUnit, Keyboard
import numpy as np
import cv2
import os
import json
import pickle
import time
from datetime import datetime


class AgentDriver:
    """
    Main controller for the autonomous driving agent.
    Supports both manual driving for data collection and AI inference.
    """
    
    def __init__(self):
        """Initialize the robot and all sensors/actuators."""
        # Initialize the robot
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Operating modes
        self.MODE_MANUAL = "manual"
        self.MODE_AI = "ai"
        self.current_mode = self.MODE_MANUAL  # Default to manual for data collection
        
        # Initialize devices
        self._init_camera()
        self._init_motors()
        self._init_sensors()
        self._init_keyboard()
        
        # Data collection setup
        self.data_folder = "collected_data"
        self._setup_data_collection()
        
        # AI model placeholder
        self.ai_model = None
        self.model_loaded = False
        
        # Control parameters
        self.max_speed = 30.0  # Maximum wheel speed
        self.max_steering_angle = 0.5  # Maximum steering angle in radians
        
        # Control state
        self.current_throttle = 0.0
        self.current_steering = 0.0
        
        # Sample counter for data collection
        self.sample_counter = 0
        
        print(f"AgentDriver initialized in {self.current_mode} mode")
    
    def _init_camera(self):
        """Initialize the front-facing camera."""
        self.camera = self.robot.getDevice("camera")
        if self.camera:
            self.camera.enable(self.timestep)
            print("Camera initialized")
        else:
            print("Warning: Camera not found")
    
    def _init_motors(self):
        """Initialize all wheel motors and steering."""
        motor_names = [
            "front_left_motor", "front_right_motor",
            "rear_left_motor", "rear_right_motor", "steering_motor"
        ]
        
        self.motors = {}
        for name in motor_names:
            motor = self.robot.getDevice(name)
            if motor:
                motor.setPosition(float('inf'))  # Set to velocity control mode
                motor.setVelocity(0.0)
                self.motors[name] = motor
                print(f"Motor {name} initialized")
            else:
                print(f"Warning: Motor {name} not found")
    
    def _init_sensors(self):
        """Initialize distance sensors and inertial unit."""
        sensor_names = ["front_sensor", "rear_sensor", "left_sensor", "right_sensor"]
        
        self.distance_sensors = {}
        for name in sensor_names:
            sensor = self.robot.getDevice(name)
            if sensor:
                sensor.enable(self.timestep)
                self.distance_sensors[name] = sensor
                print(f"Distance sensor {name} initialized")
            else:
                print(f"Warning: Distance sensor {name} not found")
        
        # Inertial unit
        self.inertial_unit = self.robot.getDevice("inertial_unit")
        if self.inertial_unit:
            self.inertial_unit.enable(self.timestep)
            print("Inertial unit initialized")
        else:
            print("Warning: Inertial unit not found")
    
    def _init_keyboard(self):
        """Initialize keyboard for manual control."""
        self.keyboard = self.robot.getKeyboard()
        if self.keyboard:
            self.keyboard.enable(self.timestep)
            print("Keyboard initialized")
        else:
            print("Warning: Keyboard not found")
    
    def _setup_data_collection(self):
        """Setup directories and files for data collection."""
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        
        # Create subdirectories
        subdirs = ["images", "actions", "sensor_data"]
        for subdir in subdirs:
            path = os.path.join(self.data_folder, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
        
        print(f"Data collection setup complete in {self.data_folder}")
    
    def get_sensor_data(self):
        """Collect data from all sensors."""
        sensor_data = {}
        
        # Camera image
        if self.camera:
            image_array = self.camera.getImageArray()
            if image_array:
                # Convert to OpenCV format (BGR)
                height = self.camera.getHeight()
                width = self.camera.getWidth()
                image = np.array(image_array, dtype=np.uint8)
                image = np.transpose(image, (1, 0, 2))  # Transpose for correct orientation
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                sensor_data['camera_image'] = image
        
        # Distance sensors
        sensor_data['distances'] = {}
        for name, sensor in self.distance_sensors.items():
            value = sensor.getValue()
            sensor_data['distances'][name] = value
        
        # Inertial unit
        if self.inertial_unit:
            roll_pitch_yaw = self.inertial_unit.getRollPitchYaw()
            sensor_data['orientation'] = {
                'roll': roll_pitch_yaw[0],
                'pitch': roll_pitch_yaw[1],
                'yaw': roll_pitch_yaw[2]
            }
        
        # Current control state
        sensor_data['control_state'] = {
            'throttle': self.current_throttle,
            'steering': self.current_steering
        }
        
        return sensor_data
    
    def process_keyboard_input(self):
        """Process keyboard input for manual control."""
        if not self.keyboard:
            return
        
        key = self.keyboard.getKey()
        
        # Throttle control
        if key == ord('W'):
            self.current_throttle = min(1.0, self.current_throttle + 0.1)
        elif key == ord('S'):
            self.current_throttle = max(-1.0, self.current_throttle - 0.1)
        else:
            # Gradual deceleration when no throttle input
            self.current_throttle *= 0.95
        
        # Steering control
        if key == ord('A'):
            self.current_steering = max(-1.0, self.current_steering - 0.1)
        elif key == ord('D'):
            self.current_steering = min(1.0, self.current_steering + 0.1)
        else:
            # Return to center when no steering input
            self.current_steering *= 0.9
        
        # Mode switching
        if key == ord('M'):
            if self.current_mode == self.MODE_MANUAL:
                if self.model_loaded:
                    self.current_mode = self.MODE_AI
                    print("Switched to AI mode")
                else:
                    print("No AI model loaded, staying in manual mode")
            else:
                self.current_mode = self.MODE_MANUAL
                print("Switched to manual mode")
        
        # Quit
        if key == ord('Q'):
            print("Exiting...")
            return False
        
        return True
    
    def apply_motor_commands(self, throttle, steering):
        """Apply throttle and steering commands to motors."""
        # Apply steering
        if "steering_motor" in self.motors:
            steering_angle = steering * self.max_steering_angle
            self.motors["steering_motor"].setPosition(steering_angle)
        
        # Calculate wheel speeds (differential drive for rear wheels)
        base_speed = throttle * self.max_speed
        steering_factor = steering * 0.3  # Reduce steering influence on speed
        
        left_speed = base_speed - steering_factor * self.max_speed
        right_speed = base_speed + steering_factor * self.max_speed
        
        # Apply to all wheels (simplified 4-wheel drive)
        if "front_left_motor" in self.motors:
            self.motors["front_left_motor"].setVelocity(left_speed)
        if "front_right_motor" in self.motors:
            self.motors["front_right_motor"].setVelocity(right_speed)
        if "rear_left_motor" in self.motors:
            self.motors["rear_left_motor"].setVelocity(left_speed)
        if "rear_right_motor" in self.motors:
            self.motors["rear_right_motor"].setVelocity(right_speed)
    
    def ai_inference(self, sensor_data):
        """Get action from AI model."""
        if not self.model_loaded or not self.ai_model:
            return 0.0, 0.0  # No action if model not loaded
        
        try:
            # Prepare observation for the model
            obs = self._prepare_observation(sensor_data)
            
            # Get action from model
            action, _ = self.ai_model.predict(obs, deterministic=True)
            
            # Extract steering and throttle from action
            steering = float(action[0])
            throttle = float(action[1])
            
            return throttle, steering
        
        except Exception as e:
            print(f"AI inference error: {e}")
            return 0.0, 0.0
    
    def _prepare_observation(self, sensor_data):
        """Prepare sensor data for AI model input."""
        # This should match the observation space defined in train_rl.py
        obs_list = []
        
        # Process camera image
        if 'camera_image' in sensor_data:
            image = sensor_data['camera_image']
            # Resize to 64x64 as expected by the model
            image_resized = cv2.resize(image, (64, 64))
            # Normalize to [0,1]
            image_normalized = image_resized.astype(np.float32) / 255.0
            obs_list.extend(image_normalized.flatten())
        else:
            # If no image, fill with zeros
            obs_list.extend(np.zeros(64*64*3))
        
        # Add distance sensor values
        if 'distances' in sensor_data:
            distances = sensor_data['distances']
            # Normalize distance values to [0,1] (assuming max range of 2.0)
            for sensor_name in ['front_sensor', 'rear_sensor', 'left_sensor', 'right_sensor']:
                value = distances.get(sensor_name, 2.0)  # Default to max range
                normalized_value = min(1.0, value / 2.0)
                obs_list.append(normalized_value)
        else:
            obs_list.extend([1.0, 1.0, 1.0, 1.0])  # Default to max distance
        
        # Add orientation data
        if 'orientation' in sensor_data:
            orientation = sensor_data['orientation']
            # Normalize angles to [-1,1]
            roll = orientation.get('roll', 0.0) / np.pi
            pitch = orientation.get('pitch', 0.0) / np.pi
            yaw = orientation.get('yaw', 0.0) / np.pi
            obs_list.extend([roll, pitch, yaw])
        else:
            obs_list.extend([0.0, 0.0, 0.0])
        
        # Add velocity estimates (simplified)
        obs_list.extend([self.current_throttle, self.current_steering])
        
        return np.array(obs_list, dtype=np.float32)
    
    def save_training_sample(self, sensor_data, action):
        """Save a training sample for imitation learning."""
        if 'camera_image' in sensor_data:
            # Save image
            timestamp = time.time()
            image_filename = f"sample_{self.sample_counter:06d}_{timestamp:.3f}.jpg"
            image_path = os.path.join(self.data_folder, "images", image_filename)
            cv2.imwrite(image_path, sensor_data['camera_image'])
            
            # Save action data
            action_data = {
                'timestamp': timestamp,
                'sample_id': self.sample_counter,
                'throttle': action[0],
                'steering': action[1],
                'image_filename': image_filename
            }
            
            action_filename = f"action_{self.sample_counter:06d}.json"
            action_path = os.path.join(self.data_folder, "actions", action_filename)
            with open(action_path, 'w') as f:
                json.dump(action_data, f, indent=2)
            
            # Save sensor data
            sensor_filename = f"sensors_{self.sample_counter:06d}.json"
            sensor_path = os.path.join(self.data_folder, "sensor_data", sensor_filename)
            
            # Prepare sensor data for JSON serialization
            save_sensor_data = {
                'timestamp': timestamp,
                'sample_id': self.sample_counter,
                'distances': sensor_data.get('distances', {}),
                'orientation': sensor_data.get('orientation', {}),
                'control_state': sensor_data.get('control_state', {})
            }
            
            with open(sensor_path, 'w') as f:
                json.dump(save_sensor_data, f, indent=2)
            
            self.sample_counter += 1
            
            if self.sample_counter % 100 == 0:
                print(f"Collected {self.sample_counter} training samples")
    
    def load_ai_model(self, model_path):
        """Load a trained AI model for inference."""
        try:
            from stable_baselines3 import PPO
            self.ai_model = PPO.load(model_path)
            self.model_loaded = True
            print(f"AI model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Failed to load AI model: {e}")
            self.model_loaded = False
    
    def run(self):
        """Main control loop."""
        print(f"Starting AgentDriver in {self.current_mode} mode")
        print("Controls: W/S=throttle, A/D=steering, M=toggle mode, Q=quit")
        
        while self.robot.step(self.timestep) != -1:
            # Process keyboard input
            if not self.process_keyboard_input():
                break
            
            # Get sensor data
            sensor_data = self.get_sensor_data()
            
            # Determine action based on current mode
            if self.current_mode == self.MODE_MANUAL:
                # Use manual control values
                throttle = self.current_throttle
                steering = self.current_steering
                
                # Save training data when manually driving
                if abs(throttle) > 0.01 or abs(steering) > 0.01:
                    self.save_training_sample(sensor_data, (throttle, steering))
                
            elif self.current_mode == self.MODE_AI:
                # Use AI model for control
                throttle, steering = self.ai_inference(sensor_data)
                self.current_throttle = throttle
                self.current_steering = steering
            
            # Apply motor commands
            self.apply_motor_commands(throttle, steering)
            
            # Display current status
            if self.robot.getTime() % 1.0 < 0.1:  # Every second
                print(f"Mode: {self.current_mode}, Throttle: {throttle:.2f}, Steering: {steering:.2f}")


# Main execution
if __name__ == "__main__":
    driver = AgentDriver()
    
    # Try to load an AI model if available
    model_path = "training_logs/final_model.zip"
    if os.path.exists(model_path):
        driver.load_ai_model(model_path)
    
    # Run the controller
    driver.run()