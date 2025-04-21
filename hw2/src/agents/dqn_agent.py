import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple

class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition', 
                                    ('state', 'action', 'next_state', 'reward', 'done'))
    
    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    """Simple Q-Network with configurable architecture."""
    
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(QNetwork, self).__init__()
        
        # For 1D state spaces (classic control environments)
        if isinstance(state_dim, int):
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim)
            )
            # Apply Kaiming initialization
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)
        
        # For image-based state spaces (Atari environments)
        else:
            self.network = nn.Sequential(
                nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * ((state_dim[1] // 8) - 3) * ((state_dim[2] // 8) - 3), 512),
                nn.ReLU(),
                nn.Linear(512, action_dim)
            )
            # Apply Kaiming initialization
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class EpsilonGreedyExploration:
    """Epsilon-greedy exploration strategy with linear decay."""
    
    def __init__(self, 
                 start_epsilon=1.0, 
                 end_epsilon=0.01, 
                 decay_steps=10000):
        self.epsilon = start_epsilon
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.step_counter = 0
    
    def select_action(self, q_values, action_space):
        """Select action using epsilon-greedy policy."""
        self.step_counter += 1
        
        # Update epsilon with linear decay
        self.epsilon = max(
            self.end_epsilon, 
            self.start_epsilon - (self.start_epsilon - self.end_epsilon) * 
            min(1.0, self.step_counter / self.decay_steps)
        )
        
        # With probability epsilon, select a random action
        if random.random() < self.epsilon:
            return action_space.sample()
        
        # Otherwise, select the action with highest Q-value
        return q_values.argmax().item()
    
    def reset(self):
        """Reset the exploration parameters."""
        self.epsilon = self.start_epsilon
        self.step_counter = 0

class BoltzmannExploration:
    """Boltzmann (softmax) exploration strategy with temperature decay."""
    
    def __init__(self, 
                 start_temp=1.0, 
                 end_temp=0.1, 
                 decay_steps=10000):
        self.temperature = start_temp
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.decay_steps = decay_steps
        self.step_counter = 0
    
    def select_action(self, q_values, action_space):
        """Select action using Boltzmann policy."""
        self.step_counter += 1
        
        # Update temperature with linear decay
        self.temperature = max(
            self.end_temp, 
            self.start_temp - (self.start_temp - self.end_temp) * 
            min(1.0, self.step_counter / self.decay_steps)
        )
        
        # Apply softmax with temperature
        probs = F.softmax(q_values / self.temperature, dim=0).cpu().detach().numpy()
        
        # Sample action according to the probability distribution
        return np.random.choice(len(probs), p=probs)
    
    def reset(self):
        """Reset the exploration parameters."""
        self.temperature = self.start_temp
        self.step_counter = 0

class ICMExploration:
    """Intrinsic Curiosity Module for exploration."""
    
    def __init__(self, state_dim, action_dim, device, 
                 feature_dim=64, 
                 intrinsic_reward_scale=0.01,
                 forward_weight=0.2):
        self.device = device
        self.action_dim = action_dim
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.forward_weight = forward_weight
        
        # Feature encoder network
        if isinstance(state_dim, int):
            self.feature_encoder = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, feature_dim)
            ).to(device)
            
            # Forward model: predicts next state features from current features and action
            self.forward_model = nn.Sequential(
                nn.Linear(feature_dim + action_dim, 128),
                nn.ReLU(),
                nn.Linear(128, feature_dim)
            ).to(device)
            
            # Inverse model: predicts action from current and next state features
            self.inverse_model = nn.Sequential(
                nn.Linear(feature_dim * 2, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            ).to(device)
        else:
            # For image inputs
            self.feature_encoder = nn.Sequential(
                nn.Conv2d(state_dim[0], 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * ((state_dim[1] // 8) - 1) * ((state_dim[2] // 8) - 1), feature_dim)
            ).to(device)
            
            # Forward model
            self.forward_model = nn.Sequential(
                nn.Linear(feature_dim + action_dim, 128),
                nn.ReLU(),
                nn.Linear(128, feature_dim)
            ).to(device)
            
            # Inverse model
            self.inverse_model = nn.Sequential(
                nn.Linear(feature_dim * 2, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.feature_encoder.parameters()) + 
            list(self.forward_model.parameters()) + 
            list(self.inverse_model.parameters()),
            lr=1e-3
        )
    
    def get_intrinsic_reward(self, state, action, next_state):
        """Compute intrinsic reward based on prediction error."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            # One-hot encode action
            action_onehot = torch.zeros(1, self.action_dim).to(self.device)
            action_onehot[0, action] = 1.0
            
            # Encode states to features
            state_features = self.feature_encoder(state_tensor)
            next_state_features = self.feature_encoder(next_state_tensor)
            
            # Predict next state features
            pred_next_features = self.forward_model(
                torch.cat([state_features, action_onehot], dim=1)
            )
            
            # Compute prediction error (intrinsic reward)
            reward = F.mse_loss(pred_next_features, next_state_features, reduction='none').sum(dim=1).item()
            return reward * self.intrinsic_reward_scale
    
    def update(self, state, action, next_state):
        """Update the ICM models."""
        # Convert inputs to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        # One-hot encode action
        action_onehot = torch.zeros(1, self.action_dim).to(self.device)
        action_onehot[0, action] = 1.0
        
        # Action as class index for inverse model
        action_tensor = torch.LongTensor([action]).to(self.device)
        
        # Encode states to features
        state_features = self.feature_encoder(state_tensor)
        next_state_features = self.feature_encoder(next_state_tensor)
        
        # Forward model loss: predict next state features
        pred_next_features = self.forward_model(
            torch.cat([state_features, action_onehot], dim=1)
        )
        forward_loss = F.mse_loss(pred_next_features, next_state_features)
        
        # Inverse model loss: predict action
        pred_action = self.inverse_model(
            torch.cat([state_features, next_state_features], dim=1)
        )
        inverse_loss = F.cross_entropy(pred_action, action_tensor)
        
        # Combined loss
        loss = self.forward_weight * forward_loss + (1 - self.forward_weight) * inverse_loss
        
        # Update models
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'forward_loss': forward_loss.item(),
            'inverse_loss': inverse_loss.item(),
            'total_loss': loss.item()
        }
    
    def select_action(self, q_values, action_space):
        """Use the base policy (typically epsilon-greedy)"""
        return q_values.argmax().item()

class RNDExploration:
    """Random Network Distillation for exploration."""
    
    def __init__(self, state_dim, device, 
                 feature_dim=64, 
                 intrinsic_reward_scale=0.1):
        self.device = device
        self.intrinsic_reward_scale = intrinsic_reward_scale
        
        # Random target network (fixed)
        if isinstance(state_dim, int):
            self.target_network = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, feature_dim)
            ).to(device)
            
            # Predictor network (trained to predict target network outputs)
            self.predictor_network = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, feature_dim)
            ).to(device)
        else:
            # For image inputs
            self.target_network = nn.Sequential(
                nn.Conv2d(state_dim[0], 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * ((state_dim[1] // 8) - 1) * ((state_dim[2] // 8) - 1), feature_dim)
            ).to(device)
            
            self.predictor_network = nn.Sequential(
                nn.Conv2d(state_dim[0], 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * ((state_dim[1] // 8) - 1) * ((state_dim[2] // 8) - 1), feature_dim)
            ).to(device)
        
        # Initialize target network with random weights and freeze
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        # Optimizer for the predictor network
        self.optimizer = optim.Adam(self.predictor_network.parameters(), lr=1e-3)
    
    def get_intrinsic_reward(self, state, action, next_state):
        """Compute intrinsic reward based on prediction error."""
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            # Get target and prediction
            target_features = self.target_network(next_state_tensor)
            predicted_features = self.predictor_network(next_state_tensor)
            
            # Compute prediction error (intrinsic reward)
            reward = F.mse_loss(predicted_features, target_features, reduction='none').sum(dim=1).item()
            return reward * self.intrinsic_reward_scale
    
    def update(self, state, action, next_state):
        """Update the predictor network."""
        # Convert inputs to tensors
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        # Get target features (no grad)
        with torch.no_grad():
            target_features = self.target_network(next_state_tensor)
        
        # Get predicted features
        predicted_features = self.predictor_network(next_state_tensor)
        
        # Compute loss
        loss = F.mse_loss(predicted_features, target_features)
        
        # Update predictor network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'rnd_loss': loss.item()}
    
    def select_action(self, q_values, action_space):
        """Use the base policy (typically epsilon-greedy)"""
        return q_values.argmax().item()

class HeuristicMountainCarAgent:
    """Heuristic-based agent for MountainCar environment.
    
    This agent uses a simple physics-based heuristic: push in the same direction as velocity.
    It's remarkably effective for MountainCar without any learning.
    """
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self, state, eval_mode=False):
        position, velocity = state
        
        # Action mapping: 0 = left, 1 = no action, 2 = right
        if velocity < 0:
            # If moving left, push left (to build momentum)
            return 0
        elif velocity > 0:
            # If moving right, push right
            return 2
        else:
            # If velocity is 0, push right (arbitrary choice to break symmetry)
            return 2
    
    def reset(self):
        pass

class DQNAgent:
    """DQN agent with various exploration strategies."""
    
    def __init__(self, state_dim, action_dim, device, 
                 exploration_type='epsilon_greedy',
                 learning_rate=1e-3,
                 gamma=0.99,
                 batch_size=64,
                 buffer_size=10000,
                 target_update_freq=100,
                 exploration_params=None):
        """Initialize DQN agent with the given parameters."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_rate = learning_rate
        self.update_counter = 0
        
        # Handle MPS device safely
        try:
            # Initialize Q networks
            self.q_network = QNetwork(state_dim, action_dim).to(device)
            self.target_network = QNetwork(state_dim, action_dim).to(device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()  # Target network is only used for inference
            
            # Initialize optimizer
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        except RuntimeError as e:
            if str(device) == "mps" and "MPS" in str(e):
                print(f"Warning: Error initializing networks on MPS device: {e}")
                print("Falling back to CPU")
                self.device = torch.device("cpu")
                self.q_network = QNetwork(state_dim, action_dim).to(self.device)
                self.target_network = QNetwork(state_dim, action_dim).to(self.device)
                self.target_network.load_state_dict(self.q_network.state_dict())
                self.target_network.eval()
                self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            else:
                raise e
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize exploration strategy
        self.exploration_type = exploration_type
        default_params = {
            'epsilon_greedy': {'start_epsilon': 1.0, 'end_epsilon': 0.01, 'decay_steps': 10000},
            'boltzmann': {'start_temp': 1.0, 'end_temp': 0.1, 'decay_steps': 10000},
            'icm': {'feature_dim': 64, 'intrinsic_reward_scale': 0.01, 'forward_weight': 0.2},
            'rnd': {'feature_dim': 64, 'intrinsic_reward_scale': 0.1}
        }
        params = exploration_params or default_params.get(exploration_type, {})
        
        if exploration_type == 'epsilon_greedy':
            self.exploration = EpsilonGreedyExploration(**params)
        elif exploration_type == 'boltzmann':
            self.exploration = BoltzmannExploration(**params)
        elif exploration_type == 'icm':
            self.exploration = ICMExploration(state_dim, action_dim, self.device, **params)
        elif exploration_type == 'rnd':
            self.exploration = RNDExploration(state_dim, self.device, **params)
        else:
            raise ValueError(f"Unknown exploration type: {exploration_type}")
    
    def act(self, state, eval_mode=False):
        """Select action based on current state and exploration strategy."""
        # Handle different input types and convert to tensor
        state_tensor = self._preprocess_state(state)
        
        # Get Q-values from network
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        # If in evaluation mode, simply take the best action
        if eval_mode:
            return q_values.argmax().item()
        
        # Otherwise, use exploration strategy
        return self.exploration.select_action(q_values, type('obj', (), {'sample': lambda: random.randint(0, self.action_dim-1)}))
    
    def _preprocess_state(self, state):
        """Convert state to tensor of appropriate format."""
        if isinstance(state, np.ndarray):
            # Handle flat arrays vs images
            if len(state.shape) == 1:
                return torch.FloatTensor(state).unsqueeze(0).to(self.device)
            elif len(state.shape) == 3:  # Image: (H, W, C) to (C, H, W)
                state = np.transpose(state, (2, 0, 1))
                return torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Handle already tensor
        if isinstance(state, torch.Tensor):
            return state.to(self.device)
        
        # Default fallback
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def update(self, state, action, next_state, reward, done):
        """Update the agent's networks based on experience."""
        # Store the transition in the replay buffer
        self.replay_buffer.push(state, action, next_state, reward, done)
        
        # If exploration uses its own model (ICM, RND), update it
        if hasattr(self.exploration, 'update'):
            self.exploration.update(state, action, next_state)
        
        # If buffer doesn't have enough samples, skip update
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Increment update counter
        self.update_counter += 1
        
        # Sample a batch of transitions
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = self.replay_buffer.Transition(*zip(*transitions))
        
        # Convert to tensors and gather into batches
        try:
            # Convert states to right format
            state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
            if len(state_batch.shape) == 4 and state_batch.shape[1] == 3:  # Image states
                # Already in pytorch format (B, C, H, W)
                pass
            
            next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
            action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
            reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
            done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)
            
            # Compute current Q values
            current_q_values = self.q_network(state_batch).gather(1, action_batch)
            
            # Compute target Q values
            with torch.no_grad():
                max_next_q_values = self.target_network(next_state_batch).max(1, keepdim=True)[0]
                target_q_values = reward_batch + (1 - done_batch) * self.gamma * max_next_q_values
            
            # Compute loss and update
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients to stabilize training
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1)
            self.optimizer.step()
            
            # Update target network periodically
            if self.update_counter % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            return {'loss': loss.item()}
            
        except RuntimeError as e:
            # Handle MPS-specific errors
            if str(self.device) == "mps" and "MPS" in str(e):
                print(f"Warning: Error in update with MPS device: {e}")
                print("Moving operations to CPU for this batch")
                
                # Try again on CPU
                self.device = torch.device("cpu")
                self.q_network = self.q_network.to(self.device)
                self.target_network = self.target_network.to(self.device)
                
                # Re-initialize optimizer
                self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
                
                # Recursive call to update now on CPU
                return self.update(state, action, next_state, reward, done)
            else:
                raise e
    
    def get_intrinsic_reward(self, state, action, next_state):
        """Get intrinsic reward from exploration model, if applicable."""
        if hasattr(self.exploration, 'get_intrinsic_reward'):
            return self.exploration.get_intrinsic_reward(state, action, next_state)
        return 0.0
    
    def save(self, path):
        """Save the agent's networks."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'exploration_type': self.exploration_type,
        }, path)
    
    def load(self, path):
        """Load the agent's networks from a saved state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # Note: Exploration state is not saved
    
    def reset(self):
        """Reset agent state for a new episode."""
        if hasattr(self.exploration, 'reset'):
            self.exploration.reset() 