import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from policy import Policy
import random

class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)
        self.fc4 = nn.Linear(128, action_dim)
        
        # Orthogonal initialization
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc4.weight, gain=0.01)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return self.fc4(x)

class DeepQNetworkPolicy(Policy):
    def __init__(self):
        # State and action dimensions
        max_stocks = 10
        max_products = 10
        stock_features = max_stocks * 3  # width, height, filled_ratio per stock
        product_features = max_products * 3  # width, height, quantity per product
        global_features = 2  # global filled ratio, step count
        self.state_dim = stock_features + product_features + global_features
        self.action_dim = max_stocks * 25  # Similar to ActorCritic

        # DQN Networks and optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )

        # Experience replay buffer
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        
        # Training parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.tau = 0.005  # For soft update
        
        # Training mode and metrics
        self.training = True
        self.steps = 0
        self.prev_filled_ratio = 0.0
        self.current_episode = []
        self.episode_rewards = []
        self.last_losses = []

    def get_action(self, observation, info):
        """Get action from policy"""
        state = self._preprocess_state(observation, info)
        state = state.to(self.device)
        self._current_state = state  # Store current state for update
        
        valid_action = None
        action_idx = None
        
        # First try epsilon-greedy
        if self.training and random.random() < self.epsilon:
            valid_action = self._get_random_valid_action(observation)
            if valid_action:
                stock_idx = valid_action["stock_idx"]
                pos_x, pos_y = valid_action["position"]
                stock = observation["stocks"][stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)
                grid_x = min(int(pos_x * 5 / stock_w), 4)
                grid_y = min(int(pos_y * 5 / stock_h), 4)
                action_idx = stock_idx * 25 + grid_x * 5 + grid_y
        
        # If epsilon-greedy failed, try Q-network
        if valid_action is None:
            with torch.no_grad():
                q_values = self.q_network(state)
                # Try all actions in order of Q-value until finding valid one
                for idx in torch.argsort(q_values, descending=True):
                    action_idx = idx.item()
                    valid_action = self._convert_action(action_idx, observation)
                    if valid_action is not None:
                        break
        
        # If still no valid action, try harder with random placement
        if valid_action is None:
            for _ in range(100):  # Increase attempts
                valid_action = self._get_random_valid_action(observation)
                if valid_action:
                    stock_idx = valid_action["stock_idx"]
                    pos_x, pos_y = valid_action["position"]
                    stock = observation["stocks"][stock_idx]
                    stock_w, stock_h = self._get_stock_size_(stock)
                    grid_x = min(int(pos_x * 5 / stock_w), 4)
                    grid_y = min(int(pos_y * 5 / stock_h), 4)
                    action_idx = stock_idx * 25 + grid_x * 5 + grid_y
                    break
        
        # If still no valid action, return a default action
        if valid_action is None:
            # Return placement at (0,0) on first available stock with smallest product
            for stock_idx, stock in enumerate(observation["stocks"]):
                for prod in sorted(observation["products"], 
                                 key=lambda x: x["size"][0] * x["size"][1]):
                    if prod["quantity"] > 0:
                        valid_action = {
                            "stock_idx": stock_idx,
                            "size": prod["size"],
                            "position": (0, 0)
                        }
                        action_idx = stock_idx * 25  # Top-left position
                        break
                if valid_action:
                    break
        
        # Calculate reward and store experience if training
        if self.training and action_idx is not None:
            reward = self.calculate_reward(valid_action, observation, info)
            self.current_episode.append((state.cpu(), action_idx, reward, None, False))
            
            if self.steps % 100 == 0:
                self._print_debug_info(valid_action, observation, info, reward)
            
            self.steps += 1
        
        return valid_action

    def update_policy(self, reward, done):
        if not self.training or len(self.memory) < self.batch_size:
            return

        # Update last experience with next state and done flag
        if len(self.current_episode) > 0:
            state, action, _, _, _ = self.current_episode[-1]
            self.current_episode[-1] = (state, action, reward, self._current_state, done)
            
            # Add episode to memory
            self.memory.extend(self.current_episode)
            if done:
                self.current_episode = []

        # Sample and prepare batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.stack([s if s is not None else torch.zeros_like(states[0]) 
                                 for s in next_states]).to(self.device)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q

        # Compute loss and update
        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft update target network
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Store loss for logging
        self.last_losses.append(loss.item())

    def calculate_reward(self, valid_action, observation, info):
        """Calculate reward using same logic as ActorCritic"""
        if valid_action is None:
            return -1.0
            
        reward = 0
        current_filled_ratio = info.get('filled_ratio', 0)
        
        # Components copied from ActorCritic
        filled_ratio_change = current_filled_ratio - self.prev_filled_ratio
        filled_ratio_reward = filled_ratio_change * 20.0
        reward += filled_ratio_reward
        
        # Add placement quality components
        stock = observation["stocks"][valid_action["stock_idx"]]
        stock_w, stock_h = self._get_stock_size_(stock)
        pos_x, pos_y = valid_action["position"]
        size_w, size_h = valid_action["size"]
        
        # Edge and corner bonuses
        if pos_x == 0 or pos_x + size_w == stock_w:
            reward += 0.5
        if pos_y == 0 or pos_y + size_h == stock_h:
            reward += 0.5
        if (pos_x == 0 or pos_x + size_w == stock_w) and \
           (pos_y == 0 or pos_y + size_h == stock_h):
            reward += 1.0
            
        # Area efficiency
        area_efficiency = (size_w * size_h) / (stock_w * stock_h)
        reward += area_efficiency * 2.0
        
        # Completion bonus
        for prod in observation["products"]:
            if prod["quantity"] == 1 and np.array_equal(prod["size"], valid_action["size"]):
                reward += 2.0
        
        self.prev_filled_ratio = current_filled_ratio
        return reward

    def _convert_action(self, action_idx, observation):
        """Convert network output to valid placement parameters"""
        max_stocks = len(observation["stocks"])
        stock_idx = min(action_idx // 25, max_stocks - 1)
        position = action_idx % 25
        pos_x = position // 5
        pos_y = position % 5
        
        # Try each product until finding valid placement
        for prod in observation["products"]:
            if prod["quantity"] > 0:
                stock = observation["stocks"][stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)
                
                # Scale position to actual stock size
                scaled_x = min(int(pos_x * stock_w / 5), stock_w - prod["size"][0])
                scaled_y = min(int(pos_y * stock_h / 5), stock_h - prod["size"][1])
                
                if self._can_place_(stock, (scaled_x, scaled_y), prod["size"]):
                    return {
                        "stock_idx": stock_idx,
                        "size": prod["size"],
                        "position": (scaled_x, scaled_y)
                    }
    
        return None

    def _print_debug_info(self, action, observation, info, reward):
        """Print detailed debug information"""
        print("\n" + "="*30 + f" Step {self.steps} Summary " + "="*30)
        print("\n1. Action Details:")
        if action is not None:
            print(f"  Stock Index: {action['stock_idx']}")
            print(f"  Position: {action['position']}")
            print(f"  Product Size: {action['size']}")
        else:
            print("  No valid action found")
        print(f"  Filled Ratio: {info['filled_ratio']:.3f}")
        print(f"  Reward: {reward:.3f}")
        
        print("\n2. Products Remaining:")
        for i, prod in enumerate(observation['products']):
            if prod['quantity'] > 0:
                print(f"  Product {i}: {prod['size']} x {prod['quantity']}")
        
        print("\n3. Training Metrics:")
        print(f"  Epsilon: {self.epsilon:.3f}")
        if len(self.last_losses) > 0:
            print(f"  Average Loss: {np.mean(self.last_losses[-100:]):.6f}")
        print("="*80 + "\n")

    def save_model(self, path):
        # Add directory if not exists
        if not os.path.dirname(path):
            path = os.path.join("saved_models", path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load_model(self, path):
        if not os.path.exists(path):
            return False
        
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        return True

    def _preprocess_state(self, observation, info):
        """Convert observation to state tensor"""
        # Get stock information
        max_stocks = 10
        stock_features = []
        for stock in observation["stocks"][:max_stocks]:
            stock_w, stock_h = self._get_stock_size_(stock)
            filled_ratio = np.sum(stock) / (stock_w * stock_h)
            stock_features.extend([stock_w, stock_h, filled_ratio])
        
        # Pad stock features if needed
        while len(stock_features) < max_stocks * 3:
            stock_features.extend([0, 0, 0])
            
        # Get product information
        max_products = 10
        product_features = []
        for prod in observation["products"][:max_products]:
            product_features.extend([*prod["size"], prod["quantity"]])
            
        # Pad product features if needed
        while len(product_features) < max_products * 3:
            product_features.extend([0, 0, 0])
            
        # Add global features
        global_features = [
            info.get('filled_ratio', 0),
            self.steps / 1000.0  # Normalized step count
        ]
        
        # Combine all features
        state = np.array(stock_features + product_features + global_features, dtype=np.float32)
        return torch.FloatTensor(state)

    def _get_stock_size_(self, stock):
        """Get width and height of stock"""
        if isinstance(stock, np.ndarray):
            return stock.shape[1], stock.shape[0]
        return stock["width"], stock["height"]

    def _can_place_(self, stock, position, size):
        """Check if product can be placed at position"""
        stock_w, stock_h = self._get_stock_size_(stock)
        pos_x, pos_y = position
        size_w, size_h = size
        
        # Check boundaries
        if pos_x < 0 or pos_y < 0 or pos_x + size_w > stock_w or pos_y + size_h > stock_h:
            return False
            
        # Check if area is empty
        if isinstance(stock, np.ndarray):
            return not np.any(stock[pos_y:pos_y + size_h, pos_x:pos_x + size_w])
        return True  # For initialization

    def _get_random_valid_action(self, observation):
        """Get a random valid action"""
        for stock_idx, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            
            for prod in observation["products"]:
                if prod["quantity"] > 0:
                    prod_w, prod_h = prod["size"]
                    
                    # Try random positions
                    for _ in range(10):
                        pos_x = np.random.randint(0, stock_w - prod_w + 1)
                        pos_y = np.random.randint(0, stock_h - prod_h + 1)
                        
                        if self._can_place_(stock, (pos_x, pos_y), prod["size"]):
                            return {
                                "stock_idx": stock_idx,
                                "size": prod["size"],
                                "position": (pos_x, pos_y)
                            }
        
        return None
