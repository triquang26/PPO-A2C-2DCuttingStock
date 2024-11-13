import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from policy import Policy

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, action_dim)
        
        # Orthogonal initialization
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
    
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        logits = self.fc3(x)
        
        if logits.size(0) == 1:
            logits = logits.squeeze(0)
            
        return logits

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 1)
        
        # Orthogonal initialization
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        value = self.fc3(x)
        
        if value.size(0) == 1:
            value = value.squeeze(0)
            
        return value

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

class ProximalPolicyOptimization(Policy):
    def __init__(self):
        super().__init__()
        # State and action dimensions
        max_stocks = 10
        max_products = 10
        stock_features = max_stocks * 3
        product_features = max_products * 3
        global_features = 2
        self.state_dim = stock_features + product_features + global_features
        self.action_dim = max_stocks * 25  # 5x5 grid for each stock
        
        # Initialize networks and optimizers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ActorNetwork(self.state_dim, self.action_dim).to(self.device)
        self.critic = CriticNetwork(self.state_dim).to(self.device)
        
        # PPO parameters
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01
        self.num_epochs = 10
        
        # Training setup
        self.memory = PPOMemory()
        self.training = True
        self.steps = 0
        self.prev_filled_ratio = 0.0
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Model saving
        self.model_path = "saved_models/"
        os.makedirs(self.model_path, exist_ok=True)
        
        # Add state normalization initialization
        self.state_mean = torch.zeros(self.state_dim).to(self.device)
        self.state_std = torch.ones(self.state_dim).to(self.device)
        
        # Add schedulers
        self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        self.critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Add these lines to store losses
        self.last_actor_loss = None
        self.last_critic_loss = None
    
    def normalize_state(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return (state - self.state_mean) / (self.state_std + 1e-8)
    
    def update_state_normalizer(self, state):
        state = torch.FloatTensor(state).to(self.device)
        self.state_mean = 0.99 * self.state_mean + 0.01 * state.mean()
        self.state_std = 0.99 * self.state_std + 0.01 * state.std()
    
    def get_action(self, observation, info):
        state = self.preprocess_observation(observation, info)
        state = state.to(self.device)
        
        with torch.no_grad():
            logits = self.actor(state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            
            if self.training:
                log_prob = dist.log_prob(action)
                value = self.critic(state)
                self.memory.add(state.cpu(), action.item(), 0, value.item(), log_prob.item(), False)
    
        action = self.convert_action(action.item(), observation)
        reward = self.calculate_reward(action, observation, info)
        
        if self.training:
            self.memory.rewards[-1] = reward
            
            # Print detailed logging every 100 steps
            if self.steps % 100 == 0:
                print("\n" + "="*30 + f" Step {self.steps} Summary " + "="*30)
                print("\n1. Action Details:")
                print(f"  Stock Index: {action['stock_idx']}")
                print(f"  Position: {action['position']}")
                print(f"  Product Size: {action['size']}")
                print(f"  Filled Ratio: {info['filled_ratio']:.3f}")
                print(f"  Reward: {reward:.3f}")
                
                print("\n2. Products Remaining:")
                for i, prod in enumerate(observation['products']):
                    if prod['quantity'] > 0:
                        print(f"  Product {i}: {prod['size']} x {prod['quantity']}")
                
                print("\n3. Training Metrics:")
                # Fix the format string issue
                actor_loss_str = f"{self.last_actor_loss:.6f}" if self.last_actor_loss is not None else "N/A"
                critic_loss_str = f"{self.last_critic_loss:.6f}" if self.last_critic_loss is not None else "N/A"
                print(f"  Actor Loss: {actor_loss_str}")
                print(f"  Critic Loss: {critic_loss_str}")
                print("="*80 + "\n")
        
        self.steps += 1
        return action
    
    def preprocess_observation(self, observation, info):
        stocks = observation["stocks"]
        products = observation["products"]
        
        # Stock features
        stock_features = []
        for stock in stocks:
            stock_w, stock_h = self._get_stock_size_(stock)
            used_space = np.sum(stock != -1)
            total_space = stock_w * stock_h
            stock_features.extend([
                stock_w / 10.0,  # Normalized width
                stock_h / 10.0,  # Normalized height
                used_space / total_space  # Utilization ratio
            ])
        
        # Product features
        prod_features = []
        for prod in products:
            if prod["quantity"] > 0:
                prod_features.extend([
                    prod["size"][0] / 10.0,  # Normalized width
                    prod["size"][1] / 10.0,  # Normalized height
                    min(prod["quantity"], 10) / 10.0  # Normalized quantity
                ])
        
        # Pad features to fixed length
        max_stocks = 10
        max_products = 10
        stock_features = stock_features[:max_stocks*3] + [0] * (max_stocks*3 - len(stock_features))
        prod_features = prod_features[:max_products*3] + [0] * (max_products*3 - len(prod_features))
        
        # Global features
        global_features = [
            info.get('filled_ratio', 0),
            self.steps / 1000.0  # Normalized step count
        ]
        
        # Combine all features
        state = np.array(stock_features + prod_features + global_features, dtype=np.float32)
        return torch.FloatTensor(state)
    
    def convert_action(self, action_idx, observation):
        # Convert network output to placement parameters
        max_stocks = len(observation["stocks"])
        stock_idx = min(action_idx // 25, max_stocks - 1)
        position = action_idx % 25
        pos_x = position // 5
        pos_y = position % 5
        
        # Find valid product placement
        valid_action = None
        for prod in observation["products"]:
            if prod["quantity"] > 0:
                stock = observation["stocks"][stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)
                
                # Scale position to actual stock size
                scaled_x = min(int(pos_x * stock_w / 5), stock_w - prod["size"][0])
                scaled_y = min(int(pos_y * stock_h / 5), stock_h - prod["size"][1])
                
                if self._can_place_(stock, (scaled_x, scaled_y), prod["size"]):
                    valid_action = {
                        "stock_idx": stock_idx,
                        "size": prod["size"],
                        "position": (scaled_x, scaled_y)
                    }
                    break
        
        # Fallback to random valid action if needed
        if valid_action is None:
            valid_action = self._get_random_valid_action(observation)
        
        return valid_action
    
    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        return torch.tensor(advantages, dtype=torch.float32).to(self.device)
    
    def update_policy(self, reward=None, done=None, info = None):
        """
        Update policy with current reward and done status
        
        Args:
            reward (float, optional): Current step reward
            done (bool, optional): Whether episode is done
        """
        # Update the last memory entry with the current reward and done status
        if reward is not None and len(self.memory.rewards) > 0:
            self.memory.rewards[-1] = reward
            self.memory.dones[-1] = done

        # Only perform PPO update when we have enough experience
        if len(self.memory.states) >= 128:  # You can adjust this batch size
            self._update_networks()
            self.memory.clear()
            
    def _update_networks(self):
        """Internal method to perform the actual PPO update"""
        if not self.training or len(self.memory.states) == 0:
            return
            
        # Convert memory to tensors
        states = torch.stack(self.memory.states).to(self.device)
        actions = torch.tensor(self.memory.actions).to(self.device)
        rewards = torch.tensor(self.memory.rewards, dtype=torch.float32).to(self.device)
        old_values = torch.tensor(self.memory.values, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(self.memory.log_probs, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.memory.dones, dtype=torch.float32).to(self.device)
        
        # Calculate advantages and returns
        advantages = self.compute_gae(rewards, old_values, dones)
        returns = advantages + old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        print(f"\nUpdating networks with {len(self.memory.states)} samples...")
        
        # PPO update loop
        for _ in range(self.num_epochs):
            # Get current policy distributions
            logits = self.actor(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Calculate policy ratio and surrogate objectives
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value function loss
            value_pred = self.critic(states)
            value_clipped = old_values + torch.clamp(
                value_pred - old_values, -self.clip_epsilon, self.clip_epsilon
            )
            value_loss_1 = (value_pred - returns).pow(2)
            value_loss_2 = (value_clipped - returns).pow(2)
            critic_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
            
            # Store the losses
            self.last_actor_loss = actor_loss.item()
            self.last_critic_loss = critic_loss.item()
            
            # Total loss
            total_loss = actor_loss + critic_loss - self.entropy_coef * entropy
            print(f"Losses - Actor: {actor_loss.item():.3f}, Critic: {critic_loss.item():.3f}")
            # Update networks
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        # Update learning rates
        mean_reward = rewards.mean().item()
        self.actor_scheduler.step(mean_reward)
        self.critic_scheduler.step(critic_loss.item())
    
    def save_model(self, filename):
        if not filename.endswith('.pt'):
            filename += '.pt'
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std,
        }, os.path.join(self.model_path, filename))
    
    def load_model(self, filename):
        try:
            checkpoint = torch.load(os.path.join(self.model_path, filename))
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.state_mean = checkpoint['state_mean']
            self.state_std = checkpoint['state_std']
            return True
        except:
            return False
    
    def calculate_reward(self, action, observation, info):
        if action is None:
            return -1.0
            
        reward = 0
        current_filled_ratio = info.get('filled_ratio', 0)
        filled_ratio_change = current_filled_ratio - self.prev_filled_ratio
        
        # Main components
        reward += filled_ratio_change * 20.0  # Filled ratio improvement
        
        # Placement quality
        stock = observation["stocks"][action["stock_idx"]]
        stock_w, stock_h = self._get_stock_size_(stock)
        pos_x, pos_y = action["position"]
        size_w, size_h = action["size"]
        
        # Edge utilization
        if pos_x == 0 or pos_x + size_w == stock_w:
            reward += 0.5
        if pos_y == 0 or pos_y + size_h == stock_h:
            reward += 0.5
            
        # Corner bonus
        if (pos_x == 0 or pos_x + size_w == stock_w) and \
           (pos_y == 0 or pos_y + size_h == stock_h):
            reward += 1.0
        
        # Area efficiency
        area_efficiency = (size_w * size_h) / (stock_w * stock_h)
        reward += area_efficiency * 2.0
        
        # Completion bonus
        for prod in observation["products"]:
            if prod["quantity"] == 1 and np.array_equal(prod["size"], action["size"]):
                reward += 2.0
                
        # Add diversity bonus
        
        # Debug logging
        # print(f"\nReward Breakdown:")
        # print(f"1. Filled Ratio Change: {filled_ratio_change * 20.0:.3f}")
        # print(f"2. Edge/Corner Bonus: {edge_bonus:.3f}")
        # print(f"3. Area Efficiency: {area_efficiency * 2.0:.3f}")
        # print(f"Total Reward: {reward:.3f}")
        
        self.prev_filled_ratio = current_filled_ratio
        return reward
    
    def _get_random_valid_action(self, observation):
        """Get a random valid action when the policy's chosen action is invalid."""
        for stock_idx, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            
            for prod in observation["products"]:
                if prod["quantity"] > 0:
                    prod_w, prod_h = prod["size"]
                    
                    # Try random positions until a valid one is found
                    for _ in range(10):  # Limit attempts to avoid infinite loop
                        pos_x = np.random.randint(0, stock_w - prod_w + 1)
                        pos_y = np.random.randint(0, stock_h - prod_h + 1)
                        
                        if self._can_place_(stock, (pos_x, pos_y), prod["size"]):
                            return {
                                "stock_idx": stock_idx,
                                "size": prod["size"],
                                "position": (pos_x, pos_y)
                            }
        
        return None  # Return None if no valid action is found