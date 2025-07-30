# Autonomous Driving Agent - Refactored Structure

This project has been refactored to support training and demonstrating four distinct autonomous driving scenarios individually: `highway-v0`, `intersection-v0`, `roundabout-v0`, and `parking-v0`.

## 📁 Project Structure

```
├── config.py                 # Shared environment configurations and utilities
├── train_highway.py          # Training script for highway-v0 scenario
├── train_intersection.py     # Training script for intersection-v0 scenario  
├── train_roundabout.py       # Training script for roundabout-v0 scenario
├── train_parking.py          # Training script for parking-v0 scenario
├── evaluate.py               # Flexible evaluation script for all scenarios
├── requirements.txt          # Updated dependencies
├── src/                      # Original custom modules (unchanged)
│   ├── environments/         # Environment wrappers
│   └── models/              # Model implementations
└── README_REFACTORED.md     # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Individual Agents

Train agents for specific scenarios:

```bash
# Train highway agent
python train_highway.py

# Train intersection agent  
python train_intersection.py

# Train roundabout agent
python train_roundabout.py

# Train parking agent
python train_parking.py
```

### 3. Demonstrate Trained Agents

Use the flexible evaluation script to watch any trained agent:

```bash
# Demonstrate highway agent
python evaluate.py highway

# Demonstrate intersection agent
python evaluate.py intersection

# Demonstrate roundabout agent  
python evaluate.py roundabout

# Demonstrate parking agent
python evaluate.py parking
```

## 📋 Detailed Usage

### Training Scripts

Each training script accepts the following arguments:

```bash
python train_<scenario>.py [OPTIONS]

Options:
  --timesteps TIMESTEPS    Number of training timesteps (default: 500,000)
  --save-path PATH         Path to save the trained model
  --help                   Show help message
```

**Examples:**
```bash
# Train with custom timesteps
python train_highway.py --timesteps 1000000

# Train with custom save path
python train_intersection.py --save-path models/my_intersection_agent.zip
```

### Evaluation Script

The evaluation script provides flexible options for demonstration:

```bash
python evaluate.py SCENARIO [OPTIONS]

Arguments:
  SCENARIO                 Scenario to demonstrate: highway, intersection, roundabout, parking

Options:
  --model PATH             Path to model file (default: auto-detect)
  --episodes N             Number of episodes to run (default: 5)
  --no-render              Run without visual rendering (faster)
  --non-deterministic      Use stochastic actions instead of deterministic
  --slow                   Add extra delay for better visualization
  --fast                   Remove delays for faster execution
  --help                   Show help message
```

**Examples:**
```bash
# Basic demonstration
python evaluate.py highway

# Demonstrate with custom model
python evaluate.py intersection --model models/custom_intersection.zip

# Run multiple episodes without rendering
python evaluate.py roundabout --episodes 10 --no-render

# Slow demonstration for detailed observation
python evaluate.py parking --slow --episodes 3
```

## 🎯 Key Features

### Modular Configuration (`config.py`)

- **Shared Settings**: All common environment configurations in one place
- **Scenario-Specific Parameters**: Customized settings for each driving scenario
- **Utility Functions**: Helper functions for environment creation and evaluation
- **No Code Duplication**: Consistent configuration across all training scripts

### Individual Training Scripts

Each `train_<scenario>.py` script:
- ✅ Imports shared settings from `config.py`
- ✅ Creates the specific highway-env environment (e.g., `intersection-v0`)
- ✅ Uses Stable-Baselines3 PPO with CnnPolicy
- ✅ Saves the trained model to a unique file (e.g., `ppo_intersection_agent.zip`)
- ✅ Provides command-line arguments for customization

### Flexible Evaluation Script

The `evaluate.py` script:
- ✅ Uses Python's `argparse` for command-line interface
- ✅ Automatically detects and loads the correct saved model
- ✅ Launches environment in `render_mode="human"` for visual demonstration
- ✅ Runs complete episodes and prints final rewards
- ✅ Works independently for all four scenarios

## 📊 Model Files

After training, you'll find the following model files:

```
models/
├── ppo_highway_agent.zip           # Final highway model
├── ppo_intersection_agent.zip      # Final intersection model  
├── ppo_roundabout_agent.zip        # Final roundabout model
├── ppo_parking_agent.zip           # Final parking model
├── ppo_highway_agent_best/         # Best highway model during training
├── ppo_intersection_agent_best/    # Best intersection model during training
├── ppo_roundabout_agent_best/      # Best roundabout model during training
└── ppo_parking_agent_best/         # Best parking model during training
```

## 🔧 Configuration Details

### Environment Settings

Each scenario has optimized parameters:

- **Highway**: High traffic, aggressive drivers, weather randomization
- **Intersection**: Moderate traffic, defensive drivers, traffic signal complexity
- **Roundabout**: Medium traffic, mixed behavior, circular navigation challenges
- **Parking**: Minimal traffic, precision maneuvering, tight space navigation

### Training Parameters

All scenarios use consistent PPO configuration:
- Learning rate: 3e-4
- Batch size: 64
- Training timesteps: 500,000 (default)
- Vectorized environments: 4 parallel environments
- Evaluation frequency: Every 10,000 steps

## 🎮 Example Workflow

1. **Train all agents:**
   ```bash
   python train_highway.py
   python train_intersection.py
   python train_roundabout.py
   python train_parking.py
   ```

2. **Test each agent:**
   ```bash
   python evaluate.py highway
   python evaluate.py intersection
   python evaluate.py roundabout  
   python evaluate.py parking
   ```

3. **Compare performance:**
   ```bash
   python evaluate.py highway --episodes 20 --no-render
   python evaluate.py intersection --episodes 20 --no-render
   ```

## 🛠️ Troubleshooting

**Model not found error:**
```bash
❌ No trained model found for intersection scenario!
   Train the model first using: python train_intersection.py
```

**Solution:** Train the agent first before trying to evaluate it.

**Dependencies missing:**
```bash
ModuleNotFoundError: No module named 'stable_baselines3'
```

**Solution:** Install requirements: `pip install -r requirements.txt`

## 📈 Original vs Refactored

| Feature | Original | Refactored |
|---------|----------|------------|
| Scenarios | Single scenario only | ✅ 4 independent scenarios |
| Configuration | Hardcoded in main script | ✅ Centralized in config.py |
| Training | One monolithic script | ✅ 4 dedicated training scripts |
| Evaluation | Complex demo script | ✅ Simple command-line evaluation |
| Model Management | Manual path handling | ✅ Automatic model detection |
| Code Reuse | Duplicated code | ✅ Shared utilities |

## 🎯 Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Train your first agent: `python train_highway.py`
3. Watch it drive: `python evaluate.py highway`
4. Train remaining agents and compare performance!

The refactored structure provides a clean, modular approach to training and evaluating autonomous driving agents across multiple scenarios while maintaining the original project's advanced features like vision processing, domain randomization, and reward shaping.