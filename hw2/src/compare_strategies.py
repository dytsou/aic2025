import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from environments.env_wrappers import create_env
from agents.dqn_agent import DQNAgent, HeuristicMountainCarAgent
from utils.train_utils import evaluate_agent, set_seed, get_device
from utils.visualization import compare_methods, create_comparison_video

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare Exploration Strategies")
    
    # Environment settings
    parser.add_argument("--env", type=str, default="mountaincar", choices=["cartpole", "mountaincar", "atari"],
                        help="Environment to use")
    parser.add_argument("--game", type=str, default="Breakout",
                        help="Atari game to use (if env is 'atari')")
    
    # Evaluation settings
    parser.add_argument("--eval_episodes", type=int, default=20,
                        help="Number of episodes for evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--load_dir", type=str, default="outputs",
                        help="Directory to load trained agents from")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="comparison",
                        help="Directory for saving comparison results")
    parser.add_argument("--create_video", action="store_true",
                        help="Create a comparison video")
    
    return parser.parse_args()

def load_trained_agent(load_path, state_dim, action_dim, exploration_type, device):
    """Load a trained agent from a checkpoint."""
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        exploration_type=exploration_type
    )
    
    try:
        agent.load(load_path)
        print(f"Loaded agent from {load_path}")
        return agent
    except Exception as e:
        print(f"Error loading agent: {e}")
        return None

def compare_exploration_strategies(args):
    """Compare different exploration strategies on the same environment."""
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Determine device (CPU/GPU)
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.env}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    if args.env == "atari":
        env_kwargs = {"game": args.game}
    else:
        env_kwargs = {}
    
    env = create_env(args.env, **env_kwargs)
    
    # Get state and action dimensions
    if args.env == "atari":
        state_dim = env.observation_space.shape
        action_dim = env.action_space.n
    else:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Define exploration strategies to compare
    exploration_strategies = ["epsilon_greedy", "boltzmann", "icm", "rnd"]
    
    # Load trained agents
    agents = {}
    
    for strategy in exploration_strategies:
        # Find the most recent saved model for this strategy
        strategy_dirs = [d for d in os.listdir(args.load_dir) 
                       if os.path.isdir(os.path.join(args.load_dir, d)) 
                       and f"{args.env}_dqn_{strategy}" in d]
        
        if strategy_dirs:
            # Use the most recent directory
            strategy_dirs.sort(reverse=True)
            agent_dir = os.path.join(args.load_dir, strategy_dirs[0])
            
            # Look for agent checkpoint
            agent_path = os.path.join(agent_dir, "agent.pt")
            if os.path.exists(agent_path):
                agent = load_trained_agent(agent_path, state_dim, action_dim, strategy, device)
                if agent:
                    agents[f"{strategy}"] = agent
            else:
                print(f"No agent checkpoint found for {strategy}")
        else:
            print(f"No directory found for {strategy}")
    
    # Add heuristic agent for MountainCar
    if args.env == "mountaincar":
        agents["heuristic"] = HeuristicMountainCarAgent(action_space=env.action_space)
        
    # Evaluate all agents
    results = {}
    
    for name, agent in agents.items():
        print(f"\nEvaluating {name}...")
        mean_reward, std_reward, episode_lengths = evaluate_agent(
            agent=agent,
            env=env,
            num_episodes=args.eval_episodes,
            render=False,
            max_steps=1000
        )
        
        results[name] = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "episode_lengths": episode_lengths,
            "success_rate": np.mean([r > -200 for r in episode_lengths])
        }
    
    # Create comparison bar plot
    plt.figure(figsize=(12, 6))
    
    # Plot mean rewards
    strategies = list(results.keys())
    mean_rewards = [results[s]["mean_reward"] for s in strategies]
    std_rewards = [results[s]["std_reward"] for s in strategies]
    
    bar_positions = np.arange(len(strategies))
    plt.bar(bar_positions, mean_rewards, yerr=std_rewards, capsize=10,
            color=['blue', 'green', 'orange', 'red', 'purple'][:len(strategies)])
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xticks(bar_positions, strategies)
    plt.xlabel("Exploration Strategy")
    plt.ylabel("Mean Reward")
    plt.title(f"Comparison of Exploration Strategies on {args.env.capitalize()}")
    
    # Add value labels
    for i, v in enumerate(mean_rewards):
        plt.text(i, v + 5, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_comparison.png"))
    plt.close()
    
    # Create success rate bar plot (for MountainCar)
    if args.env == "mountaincar":
        plt.figure(figsize=(12, 6))
        
        success_rates = [results[s]["success_rate"] * 100 for s in strategies]
        
        plt.bar(bar_positions, success_rates, 
                color=['blue', 'green', 'orange', 'red', 'purple'][:len(strategies)])
        
        plt.xticks(bar_positions, strategies)
        plt.xlabel("Exploration Strategy")
        plt.ylabel("Success Rate (%)")
        plt.title(f"Success Rate Comparison on {args.env.capitalize()}")
        
        # Add value labels
        for i, v in enumerate(success_rates):
            plt.text(i, v + 2, f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "success_rate_comparison.png"))
        plt.close()
    
    # Create episode length comparison
    plt.figure(figsize=(12, 6))
    
    mean_lengths = [np.mean(results[s]["episode_lengths"]) for s in strategies]
    std_lengths = [np.std(results[s]["episode_lengths"]) for s in strategies]
    
    plt.bar(bar_positions, mean_lengths, yerr=std_lengths, capsize=10,
            color=['blue', 'green', 'orange', 'red', 'purple'][:len(strategies)])
    
    plt.xticks(bar_positions, strategies)
    plt.xlabel("Exploration Strategy")
    plt.ylabel("Mean Episode Length")
    plt.title(f"Episode Length Comparison on {args.env.capitalize()}")
    
    # Add value labels
    for i, v in enumerate(mean_lengths):
        plt.text(i, v + 5, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "length_comparison.png"))
    plt.close()
    
    # Save detailed results
    with open(os.path.join(output_dir, "comparison_results.txt"), "w") as f:
        f.write(f"Environment: {args.env}\n")
        f.write(f"Evaluation episodes: {args.eval_episodes}\n\n")
        
        for strategy, result in results.items():
            f.write(f"Strategy: {strategy}\n")
            f.write(f"Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}\n")
            f.write(f"Mean episode length: {np.mean(result['episode_lengths']):.2f}\n")
            f.write(f"Success rate: {result['success_rate']:.2%}\n\n")
    
    # Create comparison video if requested
    if args.create_video:
        print("\nCreating comparison video...")
        # Create a new environment factory for the video
        def env_factory():
            return create_env(args.env, **env_kwargs)
        
        create_comparison_video(
            agents=agents,
            env_fn=env_factory,
            video_path=os.path.join(output_dir, "comparison_video.mp4"),
            episode_length=1000,
            fps=30
        )
    
    print(f"\nComparison results saved to {output_dir}")

if __name__ == "__main__":
    args = parse_args()
    compare_exploration_strategies(args) 