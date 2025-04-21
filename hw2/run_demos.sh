#!/bin/bash

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}RL Demo Script${NC}"
echo -e "${BLUE}==============${NC}"

# Check if outputs directory exists
if [ ! -d "outputs" ]; then
    echo -e "${RED}Error: outputs directory not found. Please train agents first.${NC}"
    exit 1
fi

# Function to find the most recent directory for a specific environment and agent
find_latest_dir() {
    local env=$1
    local agent=$2
    local exploration=$3
    
    # First check if a _latest directory exists
    latest_dir=$(find outputs -maxdepth 1 -name "${env}_${agent}_${exploration}_latest" -type d)
    
    if [ -n "$latest_dir" ]; then
        echo "$latest_dir"
        return 0
    fi
    
    # Otherwise find the most recent directory by date
    latest_dir=$(find outputs -maxdepth 1 -name "${env}_${agent}_${exploration}_*" -type d | sort -r | head -n 1)
    
    if [ -n "$latest_dir" ]; then
        echo "$latest_dir"
        return 0
    fi
    
    echo ""
    return 1
}

# Run CartPole demo
echo -e "\n${GREEN}Running demo for CartPole...${NC}"
cartpole_dir=$(find_latest_dir "cartpole" "dqn" "epsilon_greedy")

if [ -n "$cartpole_dir" ]; then
    agent_path="${cartpole_dir}/agent_final.pt"
    if [ -f "$agent_path" ]; then
        echo -e "${YELLOW}Using agent from: ${agent_path}${NC}"
        python src/record/record_classic_demo.py --env cartpole --agent dqn --agent_path "$agent_path" --episodes 1 --delay 0.05
    else
        echo -e "${RED}Agent file not found at: ${agent_path}${NC}"
        echo -e "${YELLOW}Using random agent instead${NC}"
        python src/record/record_classic_demo.py --env cartpole --agent random --episodes 1 --delay 0.05
    fi
else
    echo -e "${RED}No CartPole outputs found. Using random agent.${NC}"
    python src/record/record_classic_demo.py --env cartpole --agent random --episodes 1 --delay 0.05
fi

# Run MountainCar demo
echo -e "\n${GREEN}Running demo for MountainCar...${NC}"
mountaincar_dir=$(find_latest_dir "mountaincar" "dqn" "epsilon_greedy")

if [ -n "$mountaincar_dir" ]; then
    agent_path="${mountaincar_dir}/agent_final.pt"
    if [ -f "$agent_path" ]; then
        echo -e "${YELLOW}Using agent from: ${agent_path}${NC}"
        python src/record/record_classic_demo.py --env mountaincar --agent dqn --agent_path "$agent_path" --episodes 1 --delay 0.05
    else
        echo -e "${RED}Agent file not found at: ${agent_path}${NC}"
        echo -e "${YELLOW}Using heuristic agent instead${NC}"
        python src/record/record_classic_demo.py --env mountaincar --agent heuristic --episodes 1 --delay 0.05
    fi
else
    echo -e "${RED}No MountainCar outputs found. Using heuristic agent.${NC}"
    python src/record/record_classic_demo.py --env mountaincar --agent heuristic --episodes 1 --delay 0.05
fi

# Run Atari demo
echo -e "\n${GREEN}Running demo for Atari...${NC}"
atari_dir=$(find_latest_dir "atari" "dqn" "epsilon_greedy")

if [ -n "$atari_dir" ]; then
    agent_path="${atari_dir}/agent_final.pt"
    if [ -f "$agent_path" ]; then
        # Get the game name from directory
        game_name=$(grep -oP 'game: \K.*' "${atari_dir}/results.txt" || echo "SpaceInvaders")
        echo -e "${YELLOW}Using agent from: ${agent_path} for game: ${game_name}${NC}"
        python src/record/record_atari_demo.py --game "$game_name" --agent_path "$agent_path" --episodes 1 --delay 0.01
    else
        echo -e "${RED}Agent file not found at: ${agent_path}${NC}"
        echo -e "${YELLOW}Using random agent for SpaceInvaders instead${NC}"
        python src/record/record_atari_demo.py --game SpaceInvaders --episodes 1 --delay 0.01
    fi
else
    echo -e "${RED}No Atari outputs found. Using random agent for SpaceInvaders.${NC}"
    python src/record/record_atari_demo.py --game SpaceInvaders --episodes 1 --delay 0.01
fi

echo -e "\n${BLUE}All demos completed!${NC}"