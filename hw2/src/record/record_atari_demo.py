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

from environments.env_wrappers import create_env, AtariPreprocessing
from agents.dqn_agent import DQNAgent
from utils.train_utils import get_device

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Record Atari environment demo with DQN agent")
    
    # Environment settings
    parser.add_argument("--game", type=str, default="SpaceInvaders",
                      help="Atari game to use (e.g., SpaceInvaders, Breakout, Pong)")
    parser.add_argument("--episodes", type=int, default=3,
                      help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=5000,
                      help="Maximum steps per episode")
    parser.add_argument("--delay", type=float, default=0.01,
                      help="Delay between frames in seconds")
    parser.add_argument("--agent_path", type=str, default=None,
                      help="Path to trained agent checkpoint (if not provided, random actions will be used)")
    parser.add_argument("--render_mode", type=str, default="human",
                      help="Rendering mode (human or rgb_array)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Recording demo for Atari game: {args.game}")
    print(f"Press Ctrl+C at any time to stop an episode")
    print("=" * 50)
    
    # Create the environment
    env_kwargs = {
        "game": args.game,
        "render_mode": args.render_mode,
    }
    
    env = create_env("atari", **env_kwargs)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Initialize agent (either load from checkpoint or use random actions)
    device = get_device()
    agent = None
    
    if args.agent_path and os.path.exists(args.agent_path):
        print(f"Loading agent from {args.agent_path}")
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            exploration_type="epsilon_greedy"
        )
        agent.load(args.agent_path)
        print("Agent loaded successfully")
    else:
        print("No agent provided or could not load agent - using random actions")
    
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
                # Select action (from agent or random)
                if agent:
                    action = agent.act(obs, eval_mode=True)
                else:
                    action = env.action_space.sample()
                
                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Update totals
                episode_reward += reward
                episode_length += 1
                
                # Add delay for better visibility during recording
                if args.delay > 0:
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