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
        
        # Current action values
        self.current_steering = 0.0
        self.current_throttle = 0.0
        
        print("AgentDriver initialized successfully")
        print(f"Current mode: {self.current_mode}")
        print("Controls:")
        print("  W/S: Throttle forward/backward")
        print("  A/D: Steering left/right")
        print("  M: Toggle between manual and AI mode")
        print("  Q: Quit")
    
    def _init_camera(self):
        """Initialize the front camera."""
        self.camera = self.robot.getDevice("camera")
        if self.camera is None:
            print("Warning: Camera not found!")
            return
        
        self.camera.enable(self.timestep)
        self.camera_width = self.camera.getWidth()
        self.camera_height = self.camera.getHeight()
        print(f"Camera initialized: {self.camera_width}x{self.camera_height}")
    
    def _init_motors(self):
        """Initialize wheel motors and steering motor."""
        # Get wheel motors
        wheel_names = ["wheel_left_motor", "wheel_right_motor"]
        self.wheel_motors = []
        
        for name in wheel_names:
            motor = self.robot.getDevice(name)
            if motor is None:
                print(f"Warning: Motor {name} not found!")
                continue
            
            # Set position to infinity for velocity control
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)
            self.wheel_motors.append(motor)
        
        # Get steering motor
        self.steering_motor = self.robot.getDevice("steering_motor")
        if self.steering_motor is None:
            print("Warning: Steering motor not found!")
        else:
            self.steering_motor.setPosition(0.0)
        
        print(f"Motors initialized: {len(self.wheel_motors)} wheel motors, steering motor: {self.steering_motor is not None}")
    
    def _init_sensors(self):
        """Initialize distance sensors and inertial unit."""
        # Initialize distance sensors for parking assistance
        distance_sensor_names = ["front_sensor", "rear_sensor", "left_sensor", "right_sensor"]
        self.distance_sensors = {}
        
        for name in distance_sensor_names:
            sensor = self.robot.getDevice(name)
            if sensor is not None:
                sensor.enable(self.timestep)
                self.distance_sensors[name] = sensor
                print(f"Distance sensor {name} initialized")
            else:
                print(f"Warning: Distance sensor {name} not found")
        
        # Initialize inertial unit for heading information
        self.inertial_unit = self.robot.getDevice("inertial_unit")
        if self.inertial_unit is not None:
            self.inertial_unit.enable(self.timestep)
            print("Inertial unit initialized")
        else:
            print("Warning: Inertial unit not found")
    
    def _init_keyboard(self):
        """Initialize keyboard for manual control."""
        self.keyboard = self.robot.getKeyboard()
        if self.keyboard is not None:
            self.keyboard.enable(self.timestep)
            print("Keyboard initialized")
        else:
            print("Warning: Keyboard not found")
    
    def _setup_data_collection(self):
        """Setup directories for data collection."""
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        
        # Create subdirectories
        self.images_folder = os.path.join(self.data_folder, "images")
        self.actions_folder = os.path.join(self.data_folder, "actions")
        
        for folder in [self.images_folder, self.actions_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        # Data collection counter
        self.data_sample_count = 0
        
        # Load existing sample count if available
        count_file = os.path.join(self.data_folder, "sample_count.txt")
        if os.path.exists(count_file):
            with open(count_file, 'r') as f:
                self.data_sample_count = int(f.read().strip())
        
        print(f"Data collection setup complete. Starting from sample {self.data_sample_count}")
    
    def get_camera_image(self):
        """
        Get and preprocess the camera image.
        
        Returns:
            np.ndarray: Processed camera image
        """
        if self.camera is None:
            return None
        
        # Get raw image data
        image_data = self.camera.getImage()
        if image_data is None:
            return None
        
        # Convert to numpy array
        image = np.frombuffer(image_data, np.uint8).reshape(
            (self.camera_height, self.camera_width, 4)
        )
        
        # Convert BGRA to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        return image_rgb
    
    def get_sensor_data(self):
        """
        Get data from all sensors.
        
        Returns:
            dict: Dictionary containing all sensor readings
        """
        sensor_data = {}
        
        # Get distance sensor readings
        sensor_data['distances'] = {}
        for name, sensor in self.distance_sensors.items():
            sensor_data['distances'][name] = sensor.getValue()
        
        # Get inertial unit data (roll, pitch, yaw)
        if self.inertial_unit is not None:
            rpy = self.inertial_unit.getRollPitchYaw()
            sensor_data['orientation'] = {
                'roll': rpy[0],
                'pitch': rpy[1],
                'yaw': rpy[2]
            }
        
        # Get camera image
        sensor_data['camera_image'] = self.get_camera_image()
        
        return sensor_data
    
    def process_keyboard_input(self):
        """
        Process keyboard input for manual driving.
        
        Returns:
            tuple: (steering_angle, throttle) based on keyboard input
        """
        if self.keyboard is None:
            return 0.0, 0.0
        
        steering = 0.0
        throttle = 0.0
        
        key = self.keyboard.getKey()
        
        # Handle multiple key presses
        while key != -1:
            if key == ord('W'):  # Forward
                throttle = 0.5
            elif key == ord('S'):  # Backward
                throttle = -0.5
            elif key == ord('A'):  # Left
                steering = -0.5
            elif key == ord('D'):  # Right
                steering = 0.5
            elif key == ord('M'):  # Toggle mode
                self.toggle_mode()
            elif key == ord('Q'):  # Quit
                return None, None
            
            key = self.keyboard.getKey()
        
        return steering, throttle
    
    def toggle_mode(self):
        """Toggle between manual and AI modes."""
        if self.current_mode == self.MODE_MANUAL:
            if self.model_loaded:
                self.current_mode = self.MODE_AI
                print("Switched to AI mode")
            else:
                print("AI model not loaded. Staying in manual mode.")
        else:
            self.current_mode = self.MODE_MANUAL
            print("Switched to manual mode")
    
    def load_ai_model(self, model_path):
        """
        Load a trained AI model for inference.
        
        Args:
            model_path (str): Path to the trained model file
        """
        try:
            # This is a placeholder for model loading
            # In practice, you would load your trained CNN or RL model here
            # For example, using TensorFlow/PyTorch or Stable-Baselines3
            
            # Example for Stable-Baselines3:
            # from stable_baselines3 import PPO
            # self.ai_model = PPO.load(model_path)
            
            print(f"AI model loaded from {model_path}")
            self.model_loaded = True
            
        except Exception as e:
            print(f"Failed to load AI model: {e}")
            self.model_loaded = False
    
    def ai_inference(self, sensor_data):
        """
        Perform AI inference to get actions.
        
        Args:
            sensor_data (dict): Current sensor readings
            
        Returns:
            tuple: (steering_angle, throttle) from AI model
        """
        if not self.model_loaded or self.ai_model is None:
            return 0.0, 0.0
        
        try:
            # Preprocess the sensor data for the model
            # This is a placeholder - implement based on your model's input format
            
            # Example preprocessing for CNN:
            image = sensor_data.get('camera_image')
            if image is not None:
                # Resize and normalize image
                processed_image = cv2.resize(image, (224, 224))
                processed_image = processed_image.astype(np.float32) / 255.0
                
                # Add batch dimension
                processed_image = np.expand_dims(processed_image, axis=0)
                
                # Get AI prediction
                # action = self.ai_model.predict(processed_image)
                # steering, throttle = action[0], action[1]
                
                # Placeholder return values
                steering, throttle = 0.0, 0.1
                
                return steering, throttle
            
        except Exception as e:
            print(f"AI inference error: {e}")
            return 0.0, 0.0
        
        return 0.0, 0.0
    
    def apply_actions(self, steering_angle, throttle):
        """
        Apply steering and throttle actions to the vehicle.
        
        Args:
            steering_angle (float): Steering angle (-1 to 1)
            throttle (float): Throttle value (-1 to 1)
        """
        # Clamp values to valid ranges
        steering_angle = np.clip(steering_angle, -1.0, 1.0)
        throttle = np.clip(throttle, -1.0, 1.0)
        
        # Apply steering
        if self.steering_motor is not None:
            actual_steering = steering_angle * self.max_steering_angle
            self.steering_motor.setPosition(actual_steering)
        
        # Apply throttle to wheel motors
        wheel_speed = throttle * self.max_speed
        for motor in self.wheel_motors:
            motor.setVelocity(wheel_speed)
        
        # Store current actions
        self.current_steering = steering_angle
        self.current_throttle = throttle
    
    def save_training_sample(self, sensor_data, steering, throttle):
        """
        Save a training sample for imitation learning.
        
        Args:
            sensor_data (dict): Current sensor readings
            steering (float): Steering action
            throttle (float): Throttle action
        """
        if sensor_data.get('camera_image') is None:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sample_id = f"{self.data_sample_count:06d}_{timestamp}"
        
        # Save image
        image_path = os.path.join(self.images_folder, f"{sample_id}.jpg")
        cv2.imwrite(image_path, cv2.cvtColor(sensor_data['camera_image'], cv2.COLOR_RGB2BGR))
        
        # Save action and sensor data
        action_data = {
            'steering': steering,
            'throttle': throttle,
            'distances': sensor_data.get('distances', {}),
            'orientation': sensor_data.get('orientation', {}),
            'timestamp': timestamp,
            'sample_id': sample_id
        }
        
        action_path = os.path.join(self.actions_folder, f"{sample_id}.json")
        with open(action_path, 'w') as f:
            json.dump(action_data, f, indent=2)
        
        self.data_sample_count += 1
        
        # Update sample count file
        count_file = os.path.join(self.data_folder, "sample_count.txt")
        with open(count_file, 'w') as f:
            f.write(str(self.data_sample_count))
        
        if self.data_sample_count % 100 == 0:
            print(f"Collected {self.data_sample_count} training samples")
    
    def run(self):
        """Main execution loop for the agent driver."""
        print("Starting agent driver main loop...")
        
        # Main control loop
        while self.robot.step(self.timestep) != -1:
            # Get current sensor data
            sensor_data = self.get_sensor_data()
            
            if self.current_mode == self.MODE_MANUAL:
                # Manual driving mode for data collection
                result = self.process_keyboard_input()
                if result[0] is None:  # Quit signal
                    break
                
                steering, throttle = result
                
                # Save training sample if there's meaningful input
                if abs(steering) > 0.01 or abs(throttle) > 0.01:
                    self.save_training_sample(sensor_data, steering, throttle)
                
            else:
                # AI inference mode
                steering, throttle = self.ai_inference(sensor_data)
            
            # Apply the actions to the vehicle
            self.apply_actions(steering, throttle)
        
        print("Agent driver shutting down...")
        print(f"Total training samples collected: {self.data_sample_count}")


def main():
    """Main function to run the agent driver."""
    driver = AgentDriver()
    
    # Optionally load a pre-trained model
    # driver.load_ai_model("path/to/trained_model.zip")
    
    driver.run()


if __name__ == "__main__":
    main()