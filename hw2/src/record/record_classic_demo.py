import gymnasium as gym
import argparse
import numpy as np
import time
import os
import sys
import torch
from datetime import datetime
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from environments.env_wrappers import create_env, RewardShapingMountainCar
from agents.dqn_agent import DQNAgent, HeuristicMountainCarAgent
from utils.train_utils import get_device

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Record classic control environment demo with agent")
    
    # Environment settings
    parser.add_argument("--env", type=str, default="cartpole", choices=["cartpole", "mountaincar"],
                      help="Environment to use")
    parser.add_argument("--episodes", type=int, default=5,
                      help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=1000,
                      help="Maximum steps per episode")
    parser.add_argument("--delay", type=float, default=0.05,
                      help="Delay between frames in seconds")
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "heuristic", "random"],
                      help="Agent type to use")
    parser.add_argument("--agent_path", type=str, default=None,
                      help="Path to trained DQN agent checkpoint (if not provided, a new agent will be used)")
    parser.add_argument("--reward_shaping", action="store_true",
                      help="Use reward shaping for MountainCar")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Recording demo for environment: {args.env}")
    print(f"Using agent type: {args.agent}")
    print(f"Press Ctrl+C at any time to stop an episode")
    print("=" * 50)
    
    # Create environment
    env_kwargs = {"render_mode": "human"}
    
    # Add reward shaping for MountainCar if requested
    if args.env == "mountaincar" and args.reward_shaping:
        env_kwargs.update({
            "reward_shaping": True,
            "height_weight": 1.0,
            "velocity_weight": 0.1,
            "progress_weight": 0.5
        })
    
    env = create_env(args.env, **env_kwargs)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Initialize agent
    device = get_device()
    agent = None
    
    if args.agent == "random":
        print("Using random agent")
    elif args.agent == "heuristic" and args.env == "mountaincar":
        print("Using heuristic agent for MountainCar")
        agent = HeuristicMountainCarAgent(action_space=env.action_space)
    elif args.agent == "dqn":
        # Create a DQN agent
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            exploration_type="epsilon_greedy"
        )
        
        # Load checkpoint if provided
        if args.agent_path and os.path.exists(args.agent_path):
            print(f"Loading agent from {args.agent_path}")
            agent.load(args.agent_path)
            print("Agent loaded successfully")
        else:
            print("No agent checkpoint provided - using newly initialized DQN agent (will be random)")
    
    # Run episodes
    for episode in range(args.episodes):
        print(f"\nEpisode {episode+1}/{args.episodes}")
        print("Starting in 3 seconds... (get ready to record)")
        time.sleep(3)
        
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        try:
            for _ in range(args.max_steps):
                # Select action based on agent type
                if args.agent == "random" or agent is None:
                    action = env.action_space.sample()
                else:
                    action = agent.act(obs, eval_mode=True)
                
                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Update totals
                episode_reward += reward
                episode_length += 1
                
                # Add delay for better visibility during recording
                time.sleep(args.delay)
                
                # Stop if episode is done
                if terminated or truncated:
                    break
                    
        except KeyboardInterrupt:
            print("Episode stopped manually")
        
        print(f"Episode {episode+1} - Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    env.close()
    print("\nDemo completed!")

if __name__ == "__main__":
    main() 