import argparse
import os
import torch
import numpy as np
from datetime import datetime
import platform
import time
import shutil

from environments.env_wrappers import create_env
from agents.dqn_agent import DQNAgent, HeuristicMountainCarAgent
from utils.train_utils import train_agent, evaluate_agent, set_seed, get_device
from utils.visualization import plot_training_results, visualize_value_function

# Force CPU mode for Intel Macs to avoid MKL/MPS issues
if platform.system() == 'Darwin' and platform.machine() == 'x86_64':
    os.environ['FORCE_CPU'] = '1'
    torch.set_num_threads(4)  # Use 4 CPU threads for better performance
    print("Using CPU-optimized PyTorch on Intel Mac")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RL Exploration Strategies")
    
    # Environment settings
    parser.add_argument("--env", type=str, default="cartpole", choices=["cartpole", "mountaincar", "atari"],
                        help="Environment to use")
    parser.add_argument("--game", type=str, default="SpaceInvaders",
                        help="Atari game to use (if env is 'atari')")
    
    # Agent settings
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "heuristic"],
                        help="Agent type")
    parser.add_argument("--exploration", type=str, default="epsilon_greedy", 
                        choices=["epsilon_greedy", "boltzmann", "icm", "rnd"],
                        help="Exploration strategy")
    
    # Training settings
    parser.add_argument("--train_steps", type=int, default=50000,
                        help="Number of training steps")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of episodes for evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during evaluation")
    parser.add_argument("--render_delay", type=float, default=0.05,
                        help="Delay between frames when rendering (seconds)")
    
    # Reward shaping (for MountainCar)
    parser.add_argument("--reward_shaping", action="store_true",
                        help="Apply reward shaping for MountainCar")
    parser.add_argument("--height_weight", type=float, default=1.0,
                        help="Weight for height-based reward")
    parser.add_argument("--velocity_weight", type=float, default=0.1,
                        help="Weight for velocity-based reward")
    parser.add_argument("--progress_weight", type=float, default=0.5,
                        help="Weight for progress-based reward")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory for saving outputs")
    
    # Device settings
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force using CPU even if GPU is available")
    
    # Demo settings
    parser.add_argument("--run_demo", action="store_true",
                        help="Run a demo after training for manual recording")
    parser.add_argument("--demo_episodes", type=int, default=3,
                        help="Number of episodes to run in demo mode")
    parser.add_argument("--demo_delay", type=float, default=0.05,
                        help="Delay between frames in demo mode (seconds)")
    
    # Copy settings
    parser.add_argument("--make_latest", action="store_true",
                        help="Create a copy of the output directory without the datetime for easier reference")
    
    return parser.parse_args()

def interactive_evaluation(agent, env, num_episodes=5, render_delay=0.05, max_steps=1000):
    """
    Run an interactive evaluation with rendering for manual recording.
    
    Args:
        agent: The agent to evaluate
        env: The environment to run in
        num_episodes: Number of episodes to run
        render_delay: Delay between frames (seconds)
        max_steps: Maximum steps per episode
        
    Returns:
        Tuple of (mean_reward, std_reward, episode_lengths)
    """
    episode_rewards = []
    episode_lengths = []
    
    for i in range(num_episodes):
        print(f"\nEpisode {i+1}/{num_episodes}")
        print("Press Ctrl+C to stop the current episode")
        time.sleep(1)  # Give time to read
        
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        try:
            while not (done or truncated) and episode_length < max_steps:
                # Select action in evaluation mode
                action = agent.act(obs, eval_mode=True)
                
                # Execute action
                obs, reward, done, truncated, _ = env.step(action)
                
                # Update totals
                episode_reward += reward
                episode_length += 1
                
                # Render environment and add delay for better visibility
                env.render()
                time.sleep(render_delay)
        
        except KeyboardInterrupt:
            print("Episode stopped manually")
            done = True
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {i+1} - Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"Evaluation results - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return mean_reward, std_reward, episode_lengths

def run_demo(agent, env_name, env_kwargs, num_episodes=3, render_delay=0.05, max_steps=1000):
    """
    Run a demo of the agent in the environment for manual recording.
    
    Args:
        agent: The agent to demonstrate
        env_name: Name of the environment
        env_kwargs: Keyword arguments for environment creation
        num_episodes: Number of episodes to run
        render_delay: Delay between frames (seconds)
        max_steps: Maximum steps per episode
    """
    print("\n" + "=" * 50)
    print("DEMO MODE - Get ready to record!")
    print("=" * 50)
    
    # Make sure render_mode is set to human
    demo_env_kwargs = env_kwargs.copy()
    demo_env_kwargs["render_mode"] = "human"
    
    # Create a new environment specifically for the demo
    demo_env = create_env(env_name, **demo_env_kwargs)
    
    # Wait a moment to allow user to prepare recording
    print("\nDemo will start in 5 seconds. Prepare your screen recording software now!")
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("DEMO STARTED!")
    
    # Run the demo episodes
    for episode in range(num_episodes):
        print(f"\nDemo Episode {episode+1}/{num_episodes}")
        print("Press Ctrl+C to stop the current episode")
        
        obs, _ = demo_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        try:
            while not (done or truncated) and episode_length < max_steps:
                # Select action in evaluation mode
                action = agent.act(obs, eval_mode=True)
                
                # Execute action
                obs, reward, done, truncated, _ = demo_env.step(action)
                
                # Update totals
                episode_reward += reward
                episode_length += 1
                
                # Add delay for better visibility during recording
                time.sleep(render_delay)
                
        except KeyboardInterrupt:
            print("Episode stopped manually")
        
        print(f"Episode {episode+1} - Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    demo_env.close()
    print("\nDemo completed!")

def make_latest_copy(source_dir, env_name, agent_type, exploration_type):
    """
    Create a copy of the output directory without datetime in the name for easier reference.
    
    Args:
        source_dir: Source directory to copy
        env_name: Environment name
        agent_type: Agent type
        exploration_type: Exploration strategy type
    """
    parent_dir = os.path.dirname(source_dir)
    latest_dir = os.path.join(parent_dir, f"{env_name}_{agent_type}_{exploration_type}_latest")
    
    # Remove existing latest directory if it exists
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    
    # Copy the directory
    shutil.copytree(source_dir, latest_dir)
    print(f"Created latest copy at: {latest_dir}")
    
    return latest_dir

def main():
    """Main function to run experiments."""
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Force CPU if requested
    if args.force_cpu:
        os.environ['FORCE_CPU'] = '1'
    
    # Determine device (CPU/GPU)
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.env}_{args.agent}_{args.exploration}_{timestamp}"
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set render_mode for environment
    render_mode = "human" if args.render else None
    
    # Create environment
    if args.env == "mountaincar" and args.reward_shaping:
        env_kwargs = {
            "reward_shaping": True,
            "height_weight": args.height_weight,
            "velocity_weight": args.velocity_weight,
            "progress_weight": args.progress_weight,
            "render_mode": render_mode
        }
    elif args.env == "atari":
        env_kwargs = {
            "game": args.game,
            "render_mode": "rgb_array"
        }
    else:
        env_kwargs = {"render_mode": render_mode}
    
    env = create_env(args.env, **env_kwargs)
    
    # Create a separate eval env
    eval_kwargs = env_kwargs.copy()
    # For training we use a non-rendering env
    if not args.render:
        eval_kwargs["render_mode"] = None
    eval_env = create_env(args.env, **eval_kwargs)
    
    # Get state and action dimensions
    if args.env == "atari":
        state_dim = eval_env.observation_space.shape
        action_dim = eval_env.action_space.n
    else:
        state_dim = eval_env.observation_space.shape[0]
        action_dim = eval_env.action_space.n
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Create agent
    if args.agent == "dqn":
        # Configure exploration parameters
        if args.exploration == "epsilon_greedy":
            exploration_params = {
                "start_epsilon": 1.0,
                "end_epsilon": 0.01,
                "decay_steps": args.train_steps // 4
            }
        elif args.exploration == "boltzmann":
            exploration_params = {
                "start_temp": 10.0,
                "end_temp": 0.1,
                "decay_steps": args.train_steps // 4
            }
        elif args.exploration == "icm":
            exploration_params = {
                "feature_dim": 64,
                "intrinsic_reward_scale": 0.01,
                "forward_weight": 0.2
            }
        elif args.exploration == "rnd":
            exploration_params = {
                "feature_dim": 64,
                "intrinsic_reward_scale": 0.1
            }
        
        # Create DQN agent
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            exploration_type=args.exploration,
            exploration_params=exploration_params,
            learning_rate=1e-3,
            gamma=0.99,
            batch_size=64,
            buffer_size=50000,
            target_update_freq=500
        )
    
    elif args.agent == "heuristic":
        # Only supported for MountainCar
        if args.env != "mountaincar":
            raise ValueError("Heuristic agent is only supported for MountainCar")
        
        agent = HeuristicMountainCarAgent(action_space=env.action_space)
    
    # Train agent or run heuristic policy
    if args.agent == "dqn":
        print(f"Training {args.agent} agent with {args.exploration} exploration...")
        
        # Train agent
        training_results = train_agent(
            agent=agent,
            env=env,
            total_steps=args.train_steps,
            eval_interval=args.train_steps // 10,
            eval_episodes=args.eval_episodes // 2,  # Fewer episodes during training
            log_interval=args.train_steps // 50,
            save_path=os.path.join(output_dir, "agent.pt")
        )
        
        # Plot training results
        try:
            plot_training_results(
                results=training_results,
                title=f"{args.env.capitalize()} - {args.exploration.capitalize()} Exploration",
                save_path=os.path.join(output_dir, "training_results.png")
            )
        except Exception as e:
            print(f"Warning: Could not plot training results: {e}")
        
        # If MountainCar, visualize value function
        if args.env == "mountaincar":
            try:
                visualize_value_function(
                    q_network=agent.q_network,
                    env_id="MountainCar-v0",
                    device=device,
                    title=f"Value Function - {args.exploration.capitalize()}",
                    resolution=50,
                    save_path=os.path.join(output_dir, "value_function.png")
                )
            except Exception as e:
                print(f"Error visualizing value function: {e}")
    
    # Evaluate agent (use interactive evaluation if render is enabled)
    print(f"\nEvaluating agent performance...")
    if args.render:
        print("Running interactive evaluation (for recording)")
        mean_reward, std_reward, episode_lengths = interactive_evaluation(
            agent=agent,
            env=eval_env,
            num_episodes=args.eval_episodes,
            render_delay=args.render_delay,
            max_steps=1000
        )
    else:
        mean_reward, std_reward, episode_lengths = evaluate_agent(
            agent=agent,
            env=eval_env,
            num_episodes=args.eval_episodes,
            render=False,
            max_steps=1000
        )
    
    # Save evaluation results
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(f"Environment: {args.env}\n")
        f.write(f"Agent: {args.agent}\n")
        f.write(f"Exploration: {args.exploration}\n")
        f.write(f"Training steps: {args.train_steps}\n")
        f.write(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}\n")
        f.write(f"Mean episode length: {np.mean(episode_lengths):.2f}\n")
        f.write(f"Success rate: {np.mean([r > -200 for r in episode_lengths]):.2%}\n")
    
    print(f"Results saved to {output_dir}")
    
    # Create a copy without datetime in the name for easier reference if requested
    if args.make_latest:
        latest_dir = make_latest_copy(output_dir, args.env, args.agent, args.exploration)
        print(f"Created latest version at: {latest_dir}")
    
    # Run demo mode if requested
    if args.run_demo:
        run_demo(
            agent=agent,
            env_name=args.env,
            env_kwargs=env_kwargs,
            num_episodes=args.demo_episodes,
            render_delay=args.demo_delay,
            max_steps=1000
        )

if __name__ == "__main__":
    main() 