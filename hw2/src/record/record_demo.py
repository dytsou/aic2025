import gymnasium as gym
import argparse
import numpy as np
import time
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Record environment demo for manual screen recording")
    
    # Environment settings
    parser.add_argument("--env", type=str, default="CartPole-v1",
                      help="Environment ID to use (e.g., CartPole-v1, MountainCar-v0, ALE/SpaceInvaders-v5)")
    parser.add_argument("--episodes", type=int, default=5,
                      help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=1000,
                      help="Maximum steps per episode")
    parser.add_argument("--delay", type=float, default=0.05,
                      help="Delay between frames in seconds")
    
    # Removed --atari flag since it caused confusion
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Recording demo for environment: {args.env}")
    print(f"Press Ctrl+C at any time to stop an episode")
    print("=" * 50)
    
    # Create environment with appropriate settings
    try:
        # First try with human rendering
        env = gym.make(args.env, render_mode="human")
        print(f"Created environment with human rendering")
    except Exception as e:
        print(f"Could not create environment with human rendering: {e}")
        print("Trying with rgb_array rendering...")
        try:
            env = gym.make(args.env, render_mode="rgb_array")
            print(f"Created environment with rgb_array rendering")
        except Exception as e2:
            print(f"Could not create environment with rgb_array rendering: {e2}")
            print("Trying without specifying render mode...")
            env = gym.make(args.env)
            print(f"Created environment without specifying render mode")
    
    # Run episodes
    for episode in range(args.episodes):
        print(f"\nEpisode {episode+1}/{args.episodes}")
        print("Starting in 3 seconds... (get ready to record)")
        time.sleep(3)
        
        observation, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        try:
            for _ in range(args.max_steps):
                # Take random action
                action = env.action_space.sample()
                
                # Execute action
                observation, reward, terminated, truncated, info = env.step(action)
                
                # Update totals
                episode_reward += reward
                episode_length += 1
                
                # Ensure rendering
                try:
                    env.render()
                except Exception as e:
                    pass  # Ignore render errors
                
                # Add delay for better visibility during recording
                time.sleep(args.delay)
                
                # Stop if episode is done
                if terminated or truncated:
                    break
                    
        except KeyboardInterrupt:
            print("Episode stopped manually")
        except Exception as e:
            print(f"Error during episode: {e}")
            break
        
        print(f"Episode {episode+1} - Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    env.close()
    print("\nDemo completed!")

if __name__ == "__main__":
    main() 