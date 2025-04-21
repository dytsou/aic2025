# Deep Reinforcement Learning Exploration

This repository implements Deep Q-Networks (DQN) with various exploration strategies for reinforcement learning environments.

## Key Features

- DQN implementation with epsilon-greedy exploration
- Support for CartPole-v1, MountainCar-v0, and Atari environments
- Reward shaping for sparse reward environments
- Interactive evaluation and demo capabilities
- Comprehensive visualization tools

## Installation

1. Make the example script executable:
```bash
chmod +x run_example.sh
```

2. Run the setup script to create a virtual environment and install dependencies:
```bash
./run_example.sh
```

## Quick Start

Run experiments with different environments:

```bash
# Train DQN on CartPole
python src/main.py --env cartpole --agent dqn --exploration epsilon_greedy --train_steps 50000

# Train DQN on MountainCar with reward shaping
python src/main.py --env mountaincar --agent dqn --exploration epsilon_greedy --train_steps 50000 --reward_shaping

# Train DQN on SpaceInvaders
python src/main.py --env atari --game SpaceInvaders --agent dqn --exploration epsilon_greedy --train_steps 100000
```

## Project Structure

```
.
├── src/                    # Source code
│   ├── agents/            # RL agent implementations
│   ├── environments/      # Environment wrappers
│   └── utils/            # Utility functions
├── outputs/               # Training outputs and visualizations
├── run_example.sh        # Setup and example script
├── run_demos.sh          # Demo script for trained agents
└── requirements.txt      # Python dependencies
```

## Key Results

Our experiments show:
- CartPole-v1: Near-optimal performance with mean reward ~93.80
- MountainCar-v0: 100% success rate with reward shaping

## Interactive Evaluation

Run demos of trained agents:

```bash
# Interactive demo mode
python src/main.py --env cartpole --agent dqn --exploration epsilon_greedy --run_demo --demo_episodes 3

# Run all demos
./run_demos.sh
```

## Documentation

- [Environment Setup](run_example.sh): Environment configuration and dependency installation
- [Demo Guide](run_demos.sh): Instructions for running demonstrations

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- For Atari: ROMs must be installed separately