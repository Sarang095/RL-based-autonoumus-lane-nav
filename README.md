# 🧠 Vision-Based Autonomous Driving Agent

A comprehensive implementation of autonomous driving using **Imitation Learning** and **Reinforcement Learning** in the highway-env simulator, featuring vision-based inputs, domain randomization, and multi-agent scenarios.

## 🎯 Project Overview

This project implements a sophisticated autonomous driving agent that learns to navigate complex traffic scenarios through a two-phase approach:

1. **Phase 1: Imitation Learning (IL)** - Bootstrap learning using expert demonstrations with CNN-based behavioral cloning
2. **Phase 2: Reinforcement Learning (PPO)** - Refine and improve policies using Proximal Policy Optimization with domain randomization
3. **Phase 3: Evaluation** - Comprehensive testing across multiple scenarios with detailed analysis

### 🚗 Key Features

- **Vision-Based Learning**: Uses 84x84 grayscale images with frame stacking for temporal information
- **Domain Randomization**: Traffic density, vehicle behavior, and sensor noise variations
- **Multi-Agent Scenarios**: Realistic traffic with diverse vehicle behaviors
- **Multiple Environments**: Highway, roundabout, and parking scenarios
- **Comprehensive Evaluation**: Detailed performance analysis and model comparison

## 🛠️ Technical Stack

- **Simulator**: highway-env (2D autonomous driving environments)
- **Deep Learning**: PyTorch for neural networks
- **Reinforcement Learning**: Stable-Baselines3 (PPO implementation)
- **Computer Vision**: OpenCV for image processing
- **Visualization**: Matplotlib, TensorBoard for monitoring and analysis

## 📦 Installation

### Prerequisites

- Python 3.8+ 
- CUDA-compatible GPU (optional but recommended)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd vision-autonomous-driving
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Complete Training Pipeline

Run the full training pipeline with default settings:

```bash
python main.py --env highway --phases all
```

### Train Individual Phases

**Imitation Learning only:**
```bash
python main.py --phases il --il-episodes 200
```

**PPO only (requires trained IL model):**
```bash
python main.py --phases ppo --ppo-timesteps 1000000
```

**Evaluation only:**
```bash
python main.py --phases eval --all-scenarios --save-videos
```

### Demo Trained Models

```bash
# Interactive demo
python demo.py --interactive

# Specific scenario demo
python demo.py --scenario highway --episodes 5 --render
```

## 📋 Detailed Usage

### Training Configuration

The training process can be customized through command-line arguments:

```bash
python main.py \
    --env highway \
    --phases all \
    --il-episodes 500 \
    --ppo-timesteps 2000000 \
    --n-envs 8 \
    --eval-episodes 50 \
    --all-scenarios \
    --save-videos
```

**Key Parameters:**
- `--env`: Environment type (`highway`, `roundabout`, `parking`)
- `--il-episodes`: Number of expert demonstration episodes
- `--ppo-timesteps`: Total training steps for PPO
- `--n-envs`: Number of parallel environments for PPO
- `--eval-episodes`: Episodes per scenario for evaluation

### Advanced Configuration

Modify `src/config.py` for detailed hyperparameter tuning:

```python
# CNN Architecture
CNN_CONFIG = {
    'input_channels': 4,  # Frame stack size
    'conv_layers': [
        {'out_channels': 32, 'kernel_size': 8, 'stride': 4},
        {'out_channels': 64, 'kernel_size': 4, 'stride': 2},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1},
    ],
    'fc_layers': [512],
    'dropout_rate': 0.2,
}

# Domain Randomization
DOMAIN_RANDOMIZATION = {
    'traffic_density_range': (0.5, 2.0),
    'observation_noise_std': 0.05,
    'randomize_probability': 0.3,
}
```

## 🏗️ Architecture

### Project Structure

```
├── src/
│   ├── config.py                 # Central configuration
│   ├── environment_wrapper.py    # Environment setup and wrappers
│   ├── models/
│   │   ├── cnn_model.py          # CNN architectures
│   │   └── __init__.py
│   ├── data_collection/
│   │   ├── expert_data.py        # Expert demonstration collection
│   │   └── __init__.py
│   ├── training/
│   │   ├── imitation_learning.py # Behavioral cloning training
│   │   ├── ppo_training.py       # PPO training with custom policy
│   │   └── __init__.py
│   └── evaluation/
│       ├── evaluator.py          # Comprehensive evaluation
│       └── __init__.py
├── main.py                       # Main training pipeline
├── demo.py                       # Model demonstration
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

### CNN Architecture

The vision model uses a modified Atari DQN architecture:

```
Input: (4, 84, 84) # 4 stacked grayscale frames
Conv2D(32, 8x8, stride=4) + ReLU
Conv2D(64, 4x4, stride=2) + ReLU  
Conv2D(64, 3x3, stride=1) + ReLU
Flatten
Linear(512) + ReLU + Dropout
Linear(2) + Tanh  # [acceleration, steering]
```

### Environment Wrappers

- **VisionObservationWrapper**: Converts to grayscale, resizes to 84x84
- **FrameStack**: Stacks 4 consecutive frames for temporal information
- **DomainRandomizationWrapper**: Adds traffic/sensor variations
- **RewardShapingWrapper**: Custom reward function for better learning
- **MultiAgentWrapper**: Configures diverse vehicle behaviors

## 📊 Training Process

### Phase 1: Imitation Learning

1. **Expert Data Collection**: Gathers demonstrations using heuristic policies
2. **Behavioral Cloning**: Trains CNN to map observations → actions
3. **Validation**: Early stopping based on validation loss
4. **Evaluation**: Tests performance in clean environments

### Phase 2: Reinforcement Learning

1. **Policy Initialization**: Loads IL model weights (optional)
2. **PPO Training**: Uses custom CNN feature extractor
3. **Domain Randomization**: Trains across diverse conditions
4. **Multi-Agent Learning**: Handles realistic traffic scenarios
5. **Evaluation**: Periodic assessment during training

### Phase 3: Evaluation

1. **Scenario Testing**: Highway, roundabout, parking environments
2. **Performance Metrics**: Reward, success rate, episode length
3. **Statistical Analysis**: Mean, std, confidence intervals  
4. **Visualization**: Training curves, performance comparisons
5. **Video Recording**: Visual documentation of agent behavior

## 📈 Results and Analysis

The evaluation system provides comprehensive analysis including:

- **Performance Metrics**: Average reward, success rate, collision rate
- **Scenario Comparison**: Performance across different environments
- **Training Curves**: Loss and reward progression over time
- **Statistical Analysis**: Confidence intervals and significance tests
- **Video Documentation**: Visual verification of learned behaviors

### Expected Performance

Based on typical results:

- **Imitation Learning**: ~60-70% success rate, basic lane following
- **PPO (from scratch)**: ~75-85% success rate, improved decision making
- **PPO (IL pretrained)**: ~85-95% success rate, robust multi-scenario performance

## 🔧 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Environment not rendering**: Check display settings for headless systems
3. **Poor IL performance**: Increase expert episodes or improve heuristics
4. **PPO training slow**: Increase number of parallel environments

### Debug Mode

Enable verbose logging:
```bash
export PYTHONPATH=$PWD
python -m src.environment_wrapper  # Test environment
python -m src.models.cnn_model     # Test model
```

## 🎛️ Customization

### Adding New Scenarios

1. **Define Environment Config** in `src/config.py`:
```python
'new_scenario': {
    'observation': {'type': 'GrayscaleObservation', ...},
    'action': {'type': 'ContinuousAction'},
    # ... scenario-specific parameters
}
```

2. **Update Environment Creation** in `src/environment_wrapper.py`

3. **Add to Evaluation** in `src/evaluation/evaluator.py`

### Custom Reward Functions

Modify `RewardShapingWrapper` in `src/environment_wrapper.py`:

```python
def _shape_reward(self, obs, reward, terminated, info):
    # Add custom reward logic
    if custom_condition:
        reward += custom_bonus
    return reward
```

### Different Architectures

Implement new models in `src/models/` and update the factory function in `cnn_model.py`.

## 📚 References

- [Highway-Env Documentation](https://highway-env.readthedocs.io/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Imitation Learning Survey](https://arxiv.org/abs/1811.06711)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Highway-env team for the excellent simulation environment
- Stable-Baselines3 contributors for robust RL implementations
- PyTorch team for the deep learning framework

---

**Happy Autonomous Driving! 🚗💨**