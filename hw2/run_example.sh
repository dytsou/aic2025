#!/bin/bash
# Run examples of reinforcement learning with different exploration strategies

# Create directories
mkdir -p outputs

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Reinforcement Learning Exploration Strategies${NC}"
echo -e "${BLUE}==============================================${NC}"

# Find the correct pip command
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    echo -e "${RED}Error: pip or pip3 not found. Please install pip first.${NC}"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python is not installed. Please install Python 3 first.${NC}"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo -e "${YELLOW}Using Python command: ${PYTHON_CMD}${NC}"
echo -e "${YELLOW}Using pip command: ${PIP_CMD}${NC}"

# Detect Mac system type
if [[ "$(uname)" == "Darwin" ]]; then
    # Check if Intel Mac
    if [[ "$(uname -m)" == "x86_64" ]]; then
        echo -e "${YELLOW}Detected Intel Mac - using CPU with optimizations${NC}"
        INTEL_MAC=true
    elif [[ "$(uname -m)" == "arm64" ]]; then
        echo -e "${YELLOW}Detected Apple Silicon Mac - will use standard PyTorch${NC}"
        INTEL_MAC=false
    fi
else
    INTEL_MAC=false
fi

# Create Python virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    ${PYTHON_CMD} -m venv .venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Check if requirements are installed
echo -e "${YELLOW}Checking dependencies...${NC}"
if ! python -c "import gymnasium" &> /dev/null; then
    echo -e "${YELLOW}Installing base dependencies...${NC}"
    pip install --upgrade pip
    
    # Install NumPy 1.x to avoid compatibility warnings with PyTorch
    echo -e "${YELLOW}Installing NumPy 1.x for better compatibility...${NC}"
    pip install "numpy<2.0.0" matplotlib
    
    # Install pygame for interactive visualization
    echo -e "${YELLOW}Installing pygame for interactive rendering...${NC}"
    pip install pygame
    
    # Install PyTorch for the right platform
    if [[ "$INTEL_MAC" == true ]]; then
        echo -e "${YELLOW}Installing PyTorch for Intel Mac...${NC}"
        # Uninstall regular PyTorch if installed
        pip uninstall -y torch torchvision
        
        # Install the smallest possible version without MKL dependencies
        echo -e "${YELLOW}Installing PyTorch with minimal dependencies (no MKL)...${NC}"
        pip install torch==1.12.0 torchvision==0.13.0 --index-url https://download.pytorch.org/whl/cpu
        
        echo -e "${GREEN}PyTorch CPU version installed${NC}"
    else
        # For M1/M2 Macs or other platforms
        pip install torch torchvision
    fi
    
    # Install Gymnasium
    echo -e "${YELLOW}Installing Gymnasium...${NC}"
    pip install "gymnasium>=0.28.1"
    
    # Install Atari dependencies separately with specific versions
    echo -e "${YELLOW}Installing Atari dependencies...${NC}"
    pip install "gymnasium[atari]>=0.28.1"
    pip install ale-py==0.8.1
    pip install autorom==0.6.1
    pip install "opencv-python>=4.5.0"
    
    # Install moviepy for visualization purposes
    echo -e "${YELLOW}Installing moviepy for visualization tools...${NC}"
    pip install moviepy
    
    # Install ROMs - force accept license
    echo -e "${YELLOW}Installing Atari ROMs...${NC}"
    python -m autorom --accept-license
    
    # Verify installation
    echo -e "${YELLOW}Verifying installation...${NC}"
    if ! python -c "import gymnasium" &> /dev/null; then
        echo -e "${RED}Failed to install Gymnasium. Please check your Python environment.${NC}"
        exit 1
    fi
    
    # Verify Atari environments
    echo -e "${YELLOW}Verifying Atari environments...${NC}"
    python -c "import gymnasium as gym; env = gym.make('ALE/Pong-v5', render_mode='rgb_array'); print('Successfully created Atari environment')"
    
    echo -e "${GREEN}All dependencies installed successfully${NC}"
else
    # Check NumPy version and downgrade if needed
    NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null)
    if [[ "$NUMPY_VERSION" == 2* ]]; then
        echo -e "${YELLOW}NumPy 2.x detected. Downgrading to NumPy 1.x for better compatibility...${NC}"
        pip install "numpy<2.0.0" --force-reinstall
        echo -e "${GREEN}NumPy downgraded to version 1.x${NC}"
    fi
    
    # Check if moviepy is installed
    if ! python -c "import moviepy" &> /dev/null; then
        echo -e "${YELLOW}Installing moviepy for visualization tools...${NC}"
        pip install moviepy
    fi
fi

# Set environment variables for Intel Macs to avoid MKL errors
if [[ "$INTEL_MAC" == true ]]; then
    echo -e "${GREEN}Configuring environment for Intel Mac${NC}"
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

# Run CartPole with epsilon-greedy (short training for demo)
echo -e "\n${GREEN}Running CartPole with Epsilon-greedy Exploration (brief training)...${NC}"
python src/main.py --env cartpole --agent dqn --exploration epsilon_greedy --train_steps 10000 --eval_episodes 5 --force_cpu --render --make_latest

# Run MountainCar with heuristic agent
echo -e "\n${GREEN}Running MountainCar with Heuristic Agent...${NC}"
python src/main.py --env mountaincar --agent heuristic --eval_episodes 5 --force_cpu --render --make_latest

# Run MountainCar with DQN and reward shaping (short training for demo)
echo -e "\n${GREEN}Running MountainCar with DQN and Reward Shaping (brief training)...${NC}"
python src/main.py --env mountaincar --agent dqn --exploration epsilon_greedy --train_steps 10000 --eval_episodes 5 --reward_shaping --force_cpu --render --make_latest

# Improved Atari setup
echo -e "\n${GREEN}Setting up Atari environment...${NC}"

# Install Atari dependencies properly
echo -e "${YELLOW}Reinstalling Atari dependencies...${NC}"
pip install "gymnasium[atari]" ale-py==0.8.1 "opencv-python>=4.5.0" --upgrade
pip install autorom[accept-rom-license]==0.6.1 --upgrade

# Explicitly import ROMs using both methods
echo -e "${YELLOW}Installing Atari ROMs...${NC}"
python -m autorom --accept-license
python -m ale_py.import_roms

# Set default Atari game - use Pong as fallback
ATARI_GAME="SpaceInvaders"
GAME_WORKING=false

# Test if SpaceInvaders is available
echo -e "${YELLOW}Testing SpaceInvaders environment...${NC}"
if python -c "import gymnasium as gym; env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array'); print('SpaceInvaders environment available')" &> /dev/null; then
    GAME_WORKING=true
    echo -e "${GREEN}SpaceInvaders environment is available!${NC}"
else
    echo -e "${YELLOW}SpaceInvaders environment not available, checking alternatives...${NC}"
    
    # Try Pong as an alternative
    if python -c "import gymnasium as gym; env = gym.make('ALE/Pong-v5', render_mode='rgb_array'); print('Pong environment available')" &> /dev/null; then
        ATARI_GAME="Pong"
        GAME_WORKING=true
        echo -e "${GREEN}Using Pong environment instead${NC}"
    # Try Breakout as another alternative
    elif python -c "import gymnasium as gym; env = gym.make('ALE/Breakout-v5', render_mode='rgb_array'); print('Breakout environment available')" &> /dev/null; then
        ATARI_GAME="Breakout"
        GAME_WORKING=true
        echo -e "${GREEN}Using Breakout environment instead${NC}"
    fi
fi

# Run Atari game if available
if [ "$GAME_WORKING" = true ]; then
    echo -e "\n${BLUE}Running ${ATARI_GAME} Atari with DQN Agent...${NC}"
    echo -e "${YELLOW}This demo will run with minimal training steps for quick demonstration.${NC}"
    echo -e "${YELLOW}For better performance, consider increasing --train_steps to 100000 or more.${NC}"
    python src/main.py --env atari --game $ATARI_GAME --agent dqn --exploration epsilon_greedy --train_steps 5000 --eval_episodes 2 --force_cpu --render --render_delay 0.01 --make_latest
else
    echo -e "\n${RED}No Atari environments are available. Skipping Atari demo.${NC}"
    echo -e "${YELLOW}You can manually install Atari ROMs by running:${NC}"
    echo -e "${YELLOW}python -m autorom --accept-license${NC}"
    echo -e "${YELLOW}python -m ale_py.import_roms${NC}"
fi

echo -e "\n${BLUE}All examples completed!${NC}"
echo -e "${BLUE}Check the outputs directory for results.${NC}"
echo -e "${BLUE}For full training, increase the --train_steps parameter.${NC}"

# Deactivate virtual environment
deactivate 