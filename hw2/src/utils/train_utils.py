import numpy as np
import torch
import time
import random
import matplotlib.pyplot as plt
from collections import deque
import os
import gymnasium as gym
from typing import Dict, Any, List, Optional, Callable, Tuple
import platform

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Torch backend deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Get the available device (CPU or CUDA)."""
    # Check if FORCE_CPU environment variable is set
    if os.environ.get('FORCE_CPU') == '1':
        print("Using CPU due to FORCE_CPU environment variable")
        return torch.device("cpu")
    
    # Intel Macs should use CPU for compatibility
    if platform.system() == 'Darwin' and platform.machine() == 'x86_64':
        print("Using CPU on Intel Mac for better compatibility")
        # Set number of threads for better performance
        torch.set_num_threads(4)
        return torch.device("cpu")
    
    # For other systems, use CUDA if available
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # For M1/M2 Macs, use MPS if available
    if (platform.system() == 'Darwin' and platform.machine() == 'arm64' and 
        hasattr(torch, 'mps') and torch.backends.mps.is_available()):
        return torch.device("mps")
    
    # Default to CPU
    return torch.device("cpu")

def evaluate_agent(
    agent: Any, 
    env: gym.Env, 
    num_episodes: int = 10,
    render: bool = False,
    max_steps: int = 1000,
) -> Tuple[float, float, List[int]]:
    """
    Evaluate agent performance over multiple episodes.
    
    Args:
        agent: The agent to evaluate
        env: The environment to evaluate in
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment
        max_steps: Maximum steps per episode
        
    Returns:
        Tuple of (mean_reward, std_reward, episode_lengths)
    """
    episode_rewards = []
    episode_lengths = []
    
    for i in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated) and episode_length < max_steps:
            # Select action in evaluation mode
            action = agent.act(obs, eval_mode=True)
            
            # Execute action
            obs, reward, done, truncated, _ = env.step(action)
            
            # Update totals
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
                time.sleep(0.01)  # Small delay for smoother rendering
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Evaluation episode {i+1}/{num_episodes} - Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"Evaluation results - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return mean_reward, std_reward, episode_lengths

def train_agent(
    agent: Any,
    env: gym.Env,
    total_steps: int = 100000,
    eval_interval: int = 10000,
    eval_episodes: int = 5,
    log_interval: int = 1000,
    save_path: Optional[str] = None,
    callback: Optional[Callable] = None,
) -> Dict[str, List[float]]:
    """
    Train an agent in the given environment.
    
    Args:
        agent: The agent to train
        env: The training environment
        total_steps: Total number of environment steps to train for
        eval_interval: Steps between evaluation episodes
        eval_episodes: Number of episodes for each evaluation
        log_interval: Steps between logging
        save_path: Path to save trained agent
        callback: Optional callback function called during training
        
    Returns:
        Dictionary of training metrics
    """
    # Initialize tracking variables
    episode_rewards = []
    episode_lengths = []
    all_rewards = []
    eval_rewards = []
    losses = []
    
    # Training loop
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    
    # Moving average of rewards
    reward_window = deque(maxlen=100)
    
    for step in range(1, total_steps + 1):
        # Select action
        action = agent.act(obs)
        
        # Execute action
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Check for intrinsic rewards
        intrinsic_reward = agent.get_intrinsic_reward(obs, action, next_obs)
        total_reward = reward + intrinsic_reward
        
        # Train agent
        update_info = agent.update(obs, action, next_obs, total_reward, done)
        
        # Track episodic stats
        episode_reward += reward  # Track only external reward for evaluation
        episode_length += 1
        
        # Store loss if available
        if update_info and 'loss' in update_info:
            losses.append(update_info['loss'])
        
        # Episode termination
        if done or truncated:
            # Store episode stats
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            reward_window.append(episode_reward)
            
            # Reset environment
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
        else:
            # Continue episode
            obs = next_obs
        
        # Logging
        if step % log_interval == 0:
            mean_reward = np.mean(reward_window) if reward_window else 0
            mean_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0
            mean_loss = np.mean(losses[-100:]) if losses else 0
            
            print(f"Step {step}/{total_steps} - Mean reward: {mean_reward:.2f}, "
                  f"Mean length: {mean_length:.1f}, Mean loss: {mean_loss:.5f}")
            
            all_rewards.append(mean_reward)
        
        # Evaluation
        if step % eval_interval == 0:
            print(f"\nEvaluating agent at step {step}...")
            mean_eval_reward, _, _ = evaluate_agent(agent, env, num_episodes=eval_episodes)
            eval_rewards.append(mean_eval_reward)
            
            # Save agent if it's the best so far
            if save_path and (not eval_rewards or mean_eval_reward >= max(eval_rewards)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                agent.save(save_path)
                print(f"Saved agent checkpoint to {save_path}")
        
        # Optional callback
        if callback:
            callback(locals())
    
    # Save final agent
    if save_path:
        final_path = save_path.replace('.pt', '_final.pt')
        agent.save(final_path)
        print(f"Saved final agent to {final_path}")
    
    # Return training metrics
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'moving_avg_rewards': all_rewards,
        'eval_rewards': eval_rewards,
        'losses': losses
    }

def plot_training_results(results: Dict[str, List[float]],
                          title: str = "Training Results",
                          save_path: Optional[str] = None) -> None:
    """
    Plot training metrics.
    
    Args:
        results: Dictionary of training metrics
        title: Plot title
        save_path: Path to save the plot image
    """
    plt.figure(figsize=(12, 8))
    
    # Plot episode rewards
    if 'episode_rewards' in results:
        plt.subplot(2, 2, 1)
        plt.plot(results['episode_rewards'])
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
    
    # Plot moving average rewards
    if 'moving_avg_rewards' in results:
        plt.subplot(2, 2, 2)
        plt.plot(results['moving_avg_rewards'])
        plt.title('Moving Average Rewards (100 episodes)')
        plt.xlabel('Steps (x1000)')
        plt.ylabel('Average Reward')
    
    # Plot episode lengths
    if 'episode_lengths' in results:
        plt.subplot(2, 2, 3)
        plt.plot(results['episode_lengths'])
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
    
    # Plot evaluation rewards
    if 'eval_rewards' in results:
        plt.subplot(2, 2, 4)
        plt.plot(results['eval_rewards'])
        plt.title('Evaluation Rewards')
        plt.xlabel('Evaluation')
        plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    
    plt.show() 