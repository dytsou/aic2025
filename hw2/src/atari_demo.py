import gymnasium as gym
import numpy as np
import time
import os
import sys

def test_atari_environments():
    """
    Test if Atari environments are working properly.
    This can help detect issues with Atari dependencies.
    """
    # List of common Atari games to try with correct environment IDs
    games = [
        "ALE/SpaceInvaders-v5",
        "ALE/Breakout-v5",
        "ALE/Pong-v5",
        "ALE/MsPacman-v5"
    ]
    
    print("Testing Atari environments...")
    
    # First, check if the ale-py package is installed
    try:
        import ale_py
        print(f"ALE-Py version: {ale_py.__version__}")
    except ImportError:
        print("ALE-Py package not found. Installing now...")
        os.system("pip install ale-py==0.8.1")
        try:
            import ale_py
            print(f"ALE-Py installed, version: {ale_py.__version__}")
        except ImportError:
            print("Failed to install ALE-Py. Please install manually.")
            return False
    
    # Check if ROMs are installed
    try:
        import autorom
        print(f"AutoROM version: {autorom.__version__}")
    except ImportError:
        print("AutoROM package not found. Installing now...")
        os.system("pip install autorom==0.6.1")
        try:
            import autorom
            print(f"AutoROM installed, version: {autorom.__version__}")
            # Install ROMs automatically
            print("Installing Atari ROMs...")
            os.system("python -m autorom --accept-license")
        except ImportError:
            print("Failed to install AutoROM. Please install manually.")
            return False
    
    found_working = False
    working_env = None
    
    for env_id in games:
        try:
            print(f"Trying to create environment: {env_id}")
            env = gym.make(env_id, render_mode="rgb_array")
            
            # Test reset and step functions
            observation, info = env.reset()
            print(f"  Observation shape: {observation.shape}")
            
            # Take a random action
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            print(f"  ✓ Environment {env_id} is working!")
            print(f"  - Observation shape: {observation.shape}")
            print(f"  - Action space: {env.action_space}")
            print(f"  - Reward from random action: {reward}")
            
            found_working = True
            working_env = env_id
            env.close()
            
            # Break if we found a working environment
            break
                
        except Exception as e:
            print(f"  ✗ Error with environment {env_id}: {str(e)}")
    
    if not found_working:
        print("\nNo Atari environments could be created. Trying to fix the issue...")
        try:
            # Make sure the environment is properly registered
            print("Reinstalling key packages...")
            os.system("pip uninstall -y gymnasium ale-py autorom")
            os.system("pip install 'gymnasium[atari]>=0.28.1' ale-py==0.8.1 autorom==0.6.1")
            os.system("python -m autorom --accept-license")
            print("Packages reinstalled. Please run this script again.")
        except Exception as e:
            print(f"Failed to reinstall packages: {e}")
        
        print("\nIf the issues persist, try the following manual steps:")
        print("1. pip install 'gymnasium[atari]>=0.28.1'")
        print("2. pip install ale-py==0.8.1")
        print("3. pip install autorom==0.6.1")
        print("4. python -m autorom --accept-license")
        return False
    else:
        print(f"\nAtari environment {working_env} is working correctly!")
    
    return True, working_env

def test_atari_rendering(env_id="ALE/Breakout-v5", steps=50):
    """
    Test if Atari rendering is working properly.
    """
    try:
        # Create environment with the specified ID
        env = gym.make(env_id, render_mode="rgb_array")
        
        print(f"Testing rendering for {env_id}")
        
        # Reset environment
        observation, info = env.reset()
        
        # Run a few steps and render
        for i in range(steps):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Render and display frame info
            frame = env.render()
            
            if i % 10 == 0:  # Only print every 10 steps to avoid spam
                print(f"Step {i}: Frame shape: {frame.shape}")
            
            if terminated or truncated:
                observation, info = env.reset()
        
        print("Rendering test successful!")
        env.close()
        return True
        
    except Exception as e:
        print(f"Rendering test failed: {str(e)}")
        return False

def save_sample_frame(env_id="ALE/Breakout-v5", output_path="sample_frame.png"):
    """
    Save a sample frame from an Atari game to verify rendering.
    """
    try:
        # Create environment with the specified ID
        env = gym.make(env_id, render_mode="rgb_array")
        
        # Reset and take a few random steps
        observation, info = env.reset()
        
        for _ in range(10):  # Take 10 random actions
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
        
        # Render and save frame
        frame = env.render()
        
        if frame is not None:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(8, 6))
            plt.imshow(frame)
            plt.title(f"{env_id} Sample Frame")
            plt.axis('off')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            plt.savefig(output_path)
            plt.close()
            
            print(f"Saved sample frame to {output_path}")
            env.close()
            return True
        else:
            print("Rendering returned None")
            env.close()
            return False
            
    except Exception as e:
        print(f"Error saving sample frame: {str(e)}")
        return False

if __name__ == "__main__":
    # Create output directory for results
    os.makedirs("outputs", exist_ok=True)
    
    print("=== Atari Environment Test ===")
    
    # Test environments
    result = test_atari_environments()
    
    if isinstance(result, tuple) and result[0]:
        working_env = result[1]
        # Test rendering
        print("\n=== Atari Rendering Test ===")
        render_working = test_atari_rendering(env_id=working_env)
        
        if render_working:
            # Save a sample frame
            print("\n=== Saving Sample Frame ===")
            save_sample_frame(env_id=working_env, output_path="outputs/sample_atari_frame.png")
            print("\nAll tests passed successfully!")
    else:
        print("\nAtari environment tests failed. Please fix the issues before proceeding.") 