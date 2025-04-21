import numpy as np
import matplotlib.pyplot as plt
import pygame
from gymnasium.wrappers import RecordVideo
import os
import cv2
import torch
from typing import Dict, List, Optional, Any, Tuple
import gymnasium as gym

def plot_rewards(
    rewards: List[float],
    window_size: int = 100,
    title: str = "Episode Rewards",
    save_path: Optional[str] = None
) -> None:
    """
    Plot raw and smoothed rewards over episodes.
    
    Args:
        rewards: List of episode rewards
        window_size: Moving average window size
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    # Calculate moving average
    if len(rewards) >= window_size:
        smoothed_rewards = []
        for i in range(len(rewards) - window_size + 1):
            smoothed_rewards.append(np.mean(rewards[i:i + window_size]))
    else:
        smoothed_rewards = rewards
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, label="Raw Rewards")
    
    # Plot smoothed rewards if there are enough episodes
    if len(rewards) >= window_size:
        # Adjust x-axis for moving average (center of the window)
        offset = window_size // 2
        x_vals = list(range(offset, len(rewards) - offset + 1))
        plt.plot(x_vals, smoothed_rewards, linewidth=2, label=f"{window_size}-Episode Moving Average")
    
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    
    plt.show()

def compare_methods(
    results: Dict[str, List[float]],
    title: str = "Method Comparison",
    ylabel: str = "Average Reward",
    window_size: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Compare different methods on the same plot.
    
    Args:
        results: Dictionary mapping method names to lists of values
        title: Plot title
        ylabel: Y-axis label
        window_size: Moving average window size
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 7))
    
    for method_name, values in results.items():
        # Plot raw values with low alpha
        x_vals = list(range(len(values)))
        plt.plot(x_vals, values, alpha=0.2)
        
        # Calculate and plot smoothed values
        if len(values) >= window_size:
            smoothed_values = []
            for i in range(len(values) - window_size + 1):
                smoothed_values.append(np.mean(values[i:i + window_size]))
            
            # Adjust x-axis for moving average
            offset = window_size // 2
            smooth_x = list(range(offset, len(values) - offset + 1))
            plt.plot(smooth_x, smoothed_values, linewidth=2, label=method_name)
        else:
            plt.plot(x_vals, values, linewidth=2, label=method_name)
    
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    
    plt.show()

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
    
    plt.close()

def visualize_value_function(q_network: Any,
                            env_id: str, 
                            device: torch.device,
                            title: str = "Value Function",
                            resolution: int = 50,
                            save_path: Optional[str] = None) -> None:
    """
    Visualize the value function for a 2D environment (e.g., MountainCar).
    
    Args:
        q_network: The trained Q-network
        env_id: Environment ID
        device: Device to compute on
        title: Plot title
        resolution: Resolution of the grid
        save_path: Path to save the plot image
    """
    try:
        import gymnasium as gym
        
        # Special handling for MountainCar environment
        if "MountainCar" in env_id:
            # Create environment to get state bounds
            env = gym.make(env_id)
            
            # Get state space bounds
            low = env.observation_space.low
            high = env.observation_space.high
            
            # Create position and velocity grids
            x = np.linspace(low[0], high[0], resolution)
            y = np.linspace(low[1], high[1], resolution)
            X, Y = np.meshgrid(x, y)
            
            # Compute Q-values for each point in the grid
            q_values = np.zeros((resolution, resolution))
            
            for i in range(resolution):
                states = np.vstack([X[i], Y[i]]).T
                states_tensor = torch.FloatTensor(states).to(device)
                with torch.no_grad():
                    q_values[i] = q_network(states_tensor).max(dim=1)[0].cpu().numpy()
            
            # Plot value function
            plt.figure(figsize=(10, 8))
            
            # Main plot: Value function
            plt.pcolormesh(X, Y, q_values, cmap='viridis')
            plt.colorbar(label='Value')
            plt.xlabel('Position')
            plt.ylabel('Velocity')
            plt.title(title)
            
            # Add contour lines to highlight value landscape
            contour = plt.contour(X, Y, q_values, 10, colors='black', alpha=0.3)
            plt.clabel(contour, inline=True, fontsize=8)
            
            # Save plot if requested
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                print(f"Saved value function visualization to {save_path}")
            
            plt.close()
            env.close()
        else:
            print(f"Value function visualization not supported for {env_id}")
    
    except Exception as e:
        print(f"Error visualizing value function: {e}")

def create_comparison_video(
    agents: Dict[str, Any],
    env_fn: callable,
    video_path: str,
    episode_length: int = 1000,
    fps: int = 30
) -> None:
    """
    Create a side-by-side comparison video of different agents.
    
    Args:
        agents: Dictionary mapping agent names to agent instances
        env_fn: Function that creates a new environment instance
        video_path: Path to save the video
        episode_length: Maximum episode length
        fps: Video frame rate
    """
    try:
        import moviepy.editor as mpy
        
        # Create an environment for each agent
        envs = {name: env_fn() for name in agents}
        
        # Reset all environments
        observations = {}
        infos = {}
        for name, env in envs.items():
            observations[name], infos[name] = env.reset()
        
        # Prepare frames
        frames = []
        
        for step in range(episode_length):
            # Get actions from each agent
            actions = {}
            for name, agent in agents.items():
                actions[name] = agent.act(observations[name], eval_mode=True)
            
            # Step environments
            next_observations = {}
            rewards = {}
            terminateds = {}
            truncateds = {}
            next_infos = {}
            frames_this_step = {}
            
            all_done = True
            
            for name, env in envs.items():
                # Render current frame
                frame = env.render()
                if frame is not None:
                    frames_this_step[name] = frame
                
                # Step environment
                next_obs, reward, done, trunc, info = env.step(actions[name])
                
                observations[name] = next_obs
                rewards[name] = reward
                terminateds[name] = done
                truncateds[name] = trunc
                next_infos[name] = info
                
                all_done = all_done and (done or trunc)
            
            # Combine frames side by side
            if frames_this_step:
                combined_frame = combine_frames(frames_this_step, agent_names=list(agents.keys()))
                frames.append(combined_frame)
            
            # If all environments are done, break
            if all_done:
                break
        
        # Create video
        if frames:
            clip = mpy.ImageSequenceClip(frames, fps=fps)
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            clip.write_videofile(video_path)
            print(f"Saved comparison video to {video_path}")
        
        # Close environments
        for env in envs.values():
            env.close()
            
    except Exception as e:
        print(f"Error creating comparison video: {e}")

def combine_frames(
    frames: Dict[str, np.ndarray],
    agent_names: List[str]
) -> np.ndarray:
    """
    Combine multiple frames into a single frame with labels.
    
    Args:
        frames: Dictionary mapping agent names to frames
        agent_names: List of agent names (for ordering)
        
    Returns:
        Combined frame
    """
    # Ensure all frames have the same shape
    shapes = [frame.shape for frame in frames.values()]
    max_height = max(shape[0] for shape in shapes)
    max_width = max(shape[1] for shape in shapes)
    
    # Resize all frames to the same size
    resized_frames = {}
    for name, frame in frames.items():
        if frame.shape[:2] != (max_height, max_width):
            resized = cv2.resize(frame, (max_width, max_height))
            resized_frames[name] = resized
        else:
            resized_frames[name] = frame
    
    # Create combined frame
    n_frames = len(frames)
    total_width = max_width * n_frames
    
    if len(shapes[0]) == 3:  # Color frames (RGB)
        combined = np.zeros((max_height + 30, total_width, 3), dtype=np.uint8)
    else:  # Grayscale frames
        combined = np.zeros((max_height + 30, total_width), dtype=np.uint8)
    
    # Add frames and labels
    for i, name in enumerate(agent_names):
        if name in resized_frames:
            x_offset = i * max_width
            combined[:max_height, x_offset:x_offset + max_width] = resized_frames[name]
            
            # Add label
            label = name
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            color = (255, 255, 255)
            
            # Get text size
            text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_width, text_height = text_size
            
            # Position text
            x_text = x_offset + (max_width - text_width) // 2
            y_text = max_height + 20
            
            # Add text to image
            if len(shapes[0]) == 3:  # Color frames
                cv2.putText(combined, label, (x_text, y_text), font, font_scale, color, thickness)
            else:  # Grayscale frames (convert to color for text)
                color_frame = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
                cv2.putText(color_frame, label, (x_text, y_text), font, font_scale, color, thickness)
                combined = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    
    return combined 