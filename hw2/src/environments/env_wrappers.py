import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from collections import deque
import random
from typing import Optional, Dict, Any, Tuple

class RewardShapingMountainCar(gym.Wrapper):
    """Wrapper that adds shaped rewards to MountainCar environment."""
    
    def __init__(self, env, 
                 height_weight=1.0, 
                 velocity_weight=0.1, 
                 progress_weight=0.5,
                 potential_based=True):
        super().__init__(env)
        self.height_weight = height_weight
        self.velocity_weight = velocity_weight
        self.progress_weight = progress_weight
        self.potential_based = potential_based
        
        # For tracking progress
        self.max_position = -1.2  # Minimum position in MountainCar
        
        # For potential-based shaping
        self.previous_position = None
        self.previous_velocity = None
    
    def reset(self, **kwargs):
        """Reset the environment and progress tracking."""
        obs, info = self.env.reset(**kwargs)
        self.max_position = obs[0]  # Position is the first element
        
        # Reset potential-based tracking
        self.previous_position = obs[0]
        self.previous_velocity = obs[1]
        
        return obs, info
    
    def step(self, action):
        """Step the environment and add shaped rewards."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract state information
        position, velocity = obs
        
        # Calculate shaped rewards
        shaped_reward = 0
        
        # 1. Height-based reward: encourage gaining altitude
        height_reward = position + 0.5  # Normalized to be positive when close to the goal
        shaped_reward += self.height_weight * height_reward
        
        # 2. Velocity-based reward: encourage building momentum
        velocity_reward = abs(velocity)  # Reward higher speeds
        shaped_reward += self.velocity_weight * velocity_reward
        
        # 3. Progress-based reward: encourage exploring new states
        if position > self.max_position:
            progress_reward = position - self.max_position
            shaped_reward += self.progress_weight * progress_reward
            self.max_position = position
        else:
            progress_reward = 0
            
        # Apply potential-based shaping if enabled
        if self.potential_based:
            # Define the potential function: higher is better position and higher absolute velocity
            current_potential = position + 0.1 * abs(velocity)
            previous_potential = self.previous_position + 0.1 * abs(self.previous_velocity)
            
            # Potential-based reward: difference in potential with discount factor
            shaped_reward = 0.99 * current_potential - previous_potential
            
            # Update previous state
            self.previous_position = position
            self.previous_velocity = velocity
        
        # Original reward is -1 per step until reaching the goal
        # Keep the original reward when reaching the goal
        if reward == 0:  # Successfully reached the goal
            total_reward = 10.0  # Big bonus for reaching the goal
        else:
            total_reward = shaped_reward
            
        # Store raw and shaped rewards in info
        info['raw_reward'] = reward
        info['shaped_reward'] = shaped_reward
        info['height_reward'] = height_reward
        info['velocity_reward'] = velocity_reward
        info['progress_reward'] = progress_reward
        
        return obs, total_reward, terminated, truncated, info

class AtariPreprocessing(gym.Wrapper):
    """
    Atari preprocessing wrapper:
    - Grayscale and resize frames
    - Frame stacking
    - Episode termination on life loss
    - Reward clipping
    """
    
    def __init__(self, env, 
                 frame_skip=4,
                 frame_stack=4,
                 terminate_on_life_loss=True,
                 clip_reward=True,
                 resize_shape=(84, 84)):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.terminate_on_life_loss = terminate_on_life_loss
        self.clip_reward = clip_reward
        self.resize_shape = resize_shape
        
        # Override observation space
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(frame_stack, resize_shape[0], resize_shape[1]), 
            dtype=np.uint8
        )
        
        # Initialize frame stack
        self.frames = deque(maxlen=frame_stack)
        
        # Track lives
        self.lives = 0
    
    def reset(self, **kwargs):
        """Reset environment and initialize frame stack."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset lives - compatible with both old and new ALE versions
        try:
            if hasattr(self.env.unwrapped, 'ale') and hasattr(self.env.unwrapped.ale, 'lives'):
                self.lives = self.env.unwrapped.ale.lives()
            elif hasattr(self.env.unwrapped, 'ale') and hasattr(self.env.unwrapped.ale, 'getAvailableFunctions') and 'lives' in self.env.unwrapped.ale.getAvailableFunctions():
                self.lives = self.env.unwrapped.ale.lives()
            else:
                self.lives = 0
        except Exception:
            self.lives = 0
        
        # Clear frame stack
        self.frames.clear()
        
        # Initial frame processing
        processed_frame = self._process_frame(obs)
        
        # Fill frame stack with initial frame
        for _ in range(self.frame_stack):
            self.frames.append(processed_frame)
        
        return self._get_stacked_frames(), info
    
    def step(self, action):
        """Execute action, return preprocessed observations and modified rewards."""
        total_reward = 0
        terminated = False
        truncated = False
        
        # Skip frames and accumulate reward
        for _ in range(self.frame_skip):
            obs, reward, term, trunc, info = self.env.step(action)
            
            # Accumulate reward
            total_reward += reward
            
            # Check for termination
            terminated = terminated or term
            truncated = truncated or trunc
            
            # Check for life loss - compatible with both old and new ALE versions
            current_lives = 0
            try:
                if hasattr(self.env.unwrapped, 'ale') and hasattr(self.env.unwrapped.ale, 'lives'):
                    current_lives = self.env.unwrapped.ale.lives()
                elif hasattr(self.env.unwrapped, 'ale') and hasattr(self.env.unwrapped.ale, 'getAvailableFunctions') and 'lives' in self.env.unwrapped.ale.getAvailableFunctions():
                    current_lives = self.env.unwrapped.ale.lives()
            except Exception:
                current_lives = 0
            
            if self.terminate_on_life_loss and current_lives < self.lives:
                terminated = True
            
            self.lives = current_lives
            
            # If episode ended, don't process more frames
            if terminated or truncated:
                break
        
        # Process the final observation
        processed_frame = self._process_frame(obs)
        self.frames.append(processed_frame)
        
        # Clip reward if enabled
        if self.clip_reward:
            total_reward = np.sign(total_reward)
        
        return self._get_stacked_frames(), total_reward, terminated, truncated, info
    
    def _process_frame(self, frame):
        """Convert RGB frame to grayscale and resize."""
        import cv2
        # Convert to grayscale
        grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize
        resized = cv2.resize(grayscale, self.resize_shape, 
                             interpolation=cv2.INTER_AREA)
        
        return resized
    
    def _get_stacked_frames(self):
        """Convert frame stack to numpy array."""
        return np.array(self.frames)

class FrameStackWrapper(gym.Wrapper):
    """
    Stack consecutive frames as a single observation.
    """
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        
        # Update observation space
        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )
    
    def reset(self, **kwargs):
        """Reset environment and fill frame stack with initial observation."""
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_observation(), info
    
    def step(self, action):
        """Step environment and update frame stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self):
        """Create stacked observation from frame stack."""
        if isinstance(self.frames[0], np.ndarray):
            return np.array(self.frames)
        else:
            # Handle dictionary observations
            return np.array(list(self.frames))

def create_env(env_name, **kwargs):
    """Create and configure environments with appropriate wrappers."""
    
    # Classic control environments
    if env_name.lower() == 'cartpole':
        env = gym.make('CartPole-v1')
    
    elif env_name.lower() == 'mountaincar':
        env = gym.make('MountainCar-v0')
        
        # Apply reward shaping if specified
        if kwargs.get('reward_shaping', False):
            env = RewardShapingMountainCar(
                env,
                height_weight=kwargs.get('height_weight', 1.0),
                velocity_weight=kwargs.get('velocity_weight', 0.1),
                progress_weight=kwargs.get('progress_weight', 0.5),
                potential_based=kwargs.get('potential_based', True)
            )
    
    # Atari environments
    elif env_name.lower() == 'atari':
        # Try to create the specified Atari game
        game_name = kwargs.get('game', 'SpaceInvaders')
        render_mode = kwargs.get('render_mode', 'rgb_array')
        
        # Try different naming formats for Atari environments
        env = None
        attempted_envs = []
        
        # List of possible environment IDs to try
        env_ids = [
            f"ALE/{game_name}-v5",
            f"{game_name}-v4",
            f"ALE/{game_name}-v0",
            f"{game_name}NoFrameskip-v4"
        ]
        
        for env_id in env_ids:
            attempted_envs.append(env_id)
            try:
                env = gym.make(env_id, render_mode=render_mode)
                print(f"Successfully created Atari environment: {env_id}")
                break
            except Exception as e:
                print(f"Failed to create {env_id}: {e}")
                continue
        
        # If no environment was created, fall back to a known working environment
        if env is None:
            fallbacks = ["ALE/Breakout-v5", "Breakout-v4", "ALE/Pong-v5", "Pong-v0"]
            for fallback in fallbacks:
                try:
                    env = gym.make(fallback, render_mode=render_mode)
                    print(f"Falling back to {fallback} environment")
                    break
                except Exception:
                    continue
        
        # If still no environment, raise error
        if env is None:
            raise ValueError(f"Could not create any Atari environment. Tried: {attempted_envs + fallbacks}")
        
        # Apply Atari preprocessing
        env = AtariPreprocessing(
            env,
            frame_skip=kwargs.get('frame_skip', 4),
            frame_stack=kwargs.get('frame_stack', 4),
            terminate_on_life_loss=kwargs.get('terminate_on_life_loss', True),
            clip_reward=kwargs.get('clip_reward', True),
            resize_shape=kwargs.get('resize_shape', (84, 84))
        )
    
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    return env 