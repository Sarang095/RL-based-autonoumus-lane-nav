# Webots Autonomous Driving Project

A complete autonomous driving simulation project using Webots simulator with both Imitation Learning (IL) and Reinforcement Learning (RL) approaches. The agent learns to navigate complex scenarios including parking, roundabouts, and intersections using vision-based control.

## Project Structure

```
webots-autonomous-driving/
├── supervisor_controller.py    # Scenario management and simulation control
├── agent_driver.py            # Agent controller for data collection and AI inference
├── train_rl.py               # RL training script with PPO
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Features

- **Multi-scenario training**: Parking, intersections, and roundabouts
- **Vision-based control**: Front camera input with CNN processing
- **Sensor fusion**: Camera, distance sensors, and inertial unit data
- **Dual-mode operation**: Manual data collection and AI inference
- **PPO reinforcement learning**: Using Stable-Baselines3
- **Modular architecture**: Separate supervisor and agent controllers

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Webots World Configuration

Your Webots world file should include the following nodes with proper DEF names:

**Robot Node (Agent Car):**
```
DEF AGENT_CAR Robot {
  # Your car model with the following devices:
  # - Camera named "camera"
  # - DistanceSensors: "front_sensor", "rear_sensor", "left_sensor", "right_sensor"
  # - InertialUnit named "inertial_unit"
  # - Motors: "wheel_left_motor", "wheel_right_motor", "steering_motor"
  # - Keyboard for manual control
}
```

**Traffic Light LEDs (optional):**
```
DEF TRAFFIC_LIGHT_1 LED { ... }
DEF TRAFFIC_LIGHT_2 LED { ... }
DEF TRAFFIC_LIGHT_3 LED { ... }
```

### 3. Controller Assignment

In your Webots world:
1. Assign `supervisor_controller.py` to a Supervisor node
2. Assign `agent_driver.py` to the AGENT_CAR robot node

## Usage

### Phase 1: Data Collection (Imitation Learning)

1. **Start Webots simulation** with your world file
2. **Run the agent controller** - it will start in manual mode by default
3. **Drive manually** using keyboard controls:
   - W/S: Throttle forward/backward
   - A/D: Steering left/right
   - M: Toggle between manual and AI mode
   - Q: Quit

4. **Training data** will be automatically saved to `collected_data/` folder:
   - `images/`: Camera images from each timestep
   - `actions/`: Corresponding steering and throttle actions
   - JSON files with sensor data and metadata

### Phase 2: Reinforcement Learning Training

1. **Configure the training environment** in `train_rl.py`:
   ```python
   env = WebotsCarEnv(
       max_episode_steps=1000,
       target_scenarios=['parking', 'intersection', 'roundabout']
   )
   ```

2. **Start training**:
   ```bash
   python train_rl.py
   ```

3. **Monitor training progress**:
   - Console output shows episode rewards and statistics
   - TensorBoard logs are saved to `training_logs/`
   - View with: `tensorboard --logdir training_logs`

4. **Trained models** are saved to `training_logs/final_model.zip`

### Phase 3: AI Inference

1. **Load trained model** in `agent_driver.py`:
   ```python
   driver.load_ai_model("training_logs/final_model.zip")
   ```

2. **Switch to AI mode** by pressing 'M' during simulation

3. **Watch the agent** navigate autonomously using the trained policy

## Key Components

### Supervisor Controller (`supervisor_controller.py`)

- **Scenario Management**: Controls simulation state and environment elements
- **Agent Reset**: Repositions car to starting location for new episodes
- **Traffic Light Control**: Manages LED colors for intersection scenarios
- **Communication**: Receives commands from training scripts

Key methods:
- `reset_agent_position()`: Reset car to initial state
- `set_traffic_light_color()`: Control traffic lights
- `process_commands()`: Handle external commands

### Agent Driver (`agent_driver.py`)

- **Sensor Integration**: Camera, distance sensors, inertial unit
- **Manual Control**: Keyboard input for data collection
- **AI Inference**: Trained model prediction and action execution
- **Data Logging**: Automatic saving of training samples

Key methods:
- `get_sensor_data()`: Collect all sensor readings
- `process_keyboard_input()`: Handle manual driving
- `ai_inference()`: Get actions from trained model
- `save_training_sample()`: Save IL training data

### RL Training (`train_rl.py`)

- **Custom Gym Environment**: Webots integration with OpenAI Gym
- **PPO Training**: Using Stable-Baselines3 implementation
- **Multi-scenario Support**: Random scenario selection per episode
- **Reward Design**: Distance-based, collision avoidance, and efficiency rewards

Key methods:
- `WebotsCarEnv.step()`: Execute action and return observation
- `WebotsCarEnv.reset()`: Start new episode with random scenario
- `_calculate_reward()`: Compute reward based on performance

## Reward Structure

The RL agent is trained using a multi-component reward function:

- **Distance Progress**: +10 × improvement toward target
- **Target Reached**: +100 for successfully reaching goal
- **Collision Penalty**: -50 for hitting obstacles
- **Smooth Driving**: -0.1 × |steering_angle| for stable control
- **Forward Motion**: +0.5 × throttle for progress
- **Efficiency**: -0.01 per timestep to encourage speed

## Customization

### Adding New Scenarios

1. **Define target positions** in `train_rl.py`:
   ```python
   self.scenario_targets = {
       'new_scenario': {'target_pos': [x, y, z], 'tolerance': radius}
   }
   ```

2. **Add scenario logic** in `_reset_webots_simulation()`

3. **Update scenario list** when creating the environment

### Modifying Network Architecture

Edit the PPO model parameters in `train_rl.py`:
```python
model = PPO(
    'MlpPolicy',  # or 'CnnPolicy' for image inputs
    env,
    learning_rate=3e-4,
    # ... other parameters
)
```

### Adjusting Observation Space

Modify `_get_obs()` in `WebotsCarEnv` to include additional sensors or change image processing.

## Troubleshooting

### Common Issues

1. **"AGENT_CAR node not found"**: Ensure your robot is defined with `DEF AGENT_CAR` in the world file

2. **Device not found warnings**: Check that sensor/motor names match those in your robot model

3. **Training slow/unstable**: Adjust PPO hyperparameters or reward function weights

4. **Connection issues**: Verify Webots controllers are properly assigned and communication channels are set up

### Performance Tips

- Use GPU acceleration for training: Install CUDA-compatible PyTorch
- Reduce image resolution for faster processing
- Adjust `max_episode_steps` based on scenario complexity
- Monitor training with TensorBoard for hyperparameter tuning

## Dependencies

See `requirements.txt` for the complete list. Key dependencies:
- **Webots**: R2023a or later (simulation environment)
- **Stable-Baselines3**: RL algorithms implementation
- **OpenCV**: Image processing
- **NumPy**: Numerical computations
- **PyTorch**: Deep learning backend

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to extend this project with:
- Additional scenarios (highway driving, parking lots, etc.)
- Advanced neural network architectures
- Multi-agent scenarios
- Real-world sensor simulation improvements