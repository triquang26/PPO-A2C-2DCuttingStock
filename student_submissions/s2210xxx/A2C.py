import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from policy import Policy
# ===========================
# Base Policy Class (Placeholder)
# ===========================
class CuttingStockMetrics:
    def __init__(self):
        # Episode-level metrics
        self.episode_metrics = {
            'filled_ratios': [],
            'waste_ratios': [],
            'completed_products': [],
            'invalid_actions': [],
            'episode_lengths': [],
            'rewards': [],
            'edge_utilization': [],
            'corner_placements': [],
            'largest_waste_area': [],
            'product_completion_order': []
        }
        
        # Best scores
        self.best_scores = {
            'best_filled_ratio': 0.0,
            'best_reward': float('-inf'),
            'best_episode': -1
        }
        
        # Running averages
        self.running_averages = {
            'filled_ratio': deque(maxlen=10),
            'waste_ratio': deque(maxlen=10),
            'reward': deque(maxlen=10)
        }

class EpisodeEvaluator:
    def __init__(self):
        self.metrics = {
            'episode_number': 0,
            'steps': 0,
            'filled_ratio': 0.0,
            'total_reward': 0.0,
            'total_waste': 0.0,
            'waste_per_stock': 0.0,
            'num_stocks_used': 0
        }
    
    def calculate_waste(self, observation):
        """Calculate waste for all used stocks"""
        total_waste = 0
        used_stocks = 0
        
        for stock in observation['stocks']:
            if np.any(stock > 0):  # Stock has been used
                stock_area = stock.shape[0] * stock.shape[1]
                used_area = np.sum(stock > 0)
                waste = stock_area - used_area
                total_waste += waste
                used_stocks += 1
                
        return {
            'total_waste': total_waste,
            'num_stocks': used_stocks,
            'waste_per_stock': total_waste / used_stocks if used_stocks > 0 else 0
        }

    def evaluate_episode(self, observation, info, episode_data):
        """Calculate comprehensive episode quality score"""
        waste_metrics = self.calculate_waste(observation)
        
        self.metrics.update({
            'episode_number': episode_data['episode_number'],
            'steps': episode_data['steps'],
            'filled_ratio': info['filled_ratio'],
            'total_reward': episode_data['total_reward'],
            'total_waste': waste_metrics['total_waste'],
            'waste_per_stock': waste_metrics['waste_per_stock'],
            'num_stocks_used': waste_metrics['num_stocks']
        })
        
        return self.get_summary()
    
    def get_summary(self):
        """Return formatted summary of episode performance"""
        summary = f"\n{'='*20} Episode {self.metrics['episode_number']} Quality Report {'='*20}\n"
        summary += f"Steps: {self.metrics['steps']}\n"
        summary += f"Filled Ratio: {self.metrics['filled_ratio']:.3f}\n"
        summary += f"Total Waste: {self.metrics['total_waste']}\n"
        summary += f"Number of Stocks Used: {self.metrics['num_stocks_used']}\n"
        summary += f"Waste per Stock: {self.metrics['waste_per_stock']:.1f}\n"
        summary += f"Total Reward: {self.metrics['total_reward']:.2f}\n"
        summary += "="*70
        return summary

# ===========================
# Neural Network Definitions
# ===========================
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
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),  # Giảm từ 256 xuống 128
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),  # Thêm dropout để giảm overfitting
            nn.Linear(128, 32),   # Giảm từ 64 xuống 32
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize weights với gain nhỏ hơn
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)  # Giảm gain xuống 0.01
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.network(state)

# ===========================
# Memory for A2C
# ===========================
class A2CMemory:
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

# ===========================
# Advantage Actor-Critic (A2C) Class
# ===========================
class AdvantageActorCritic(Policy):
    def __init__(self):
        super().__init__()
        # State and action dimensions
        max_stocks = 100
        self.max_products = None
        self.state_dim = None
        self.action_dim = max_stocks * 25
        
        # Training flag
        self.training = True
        
        # Initialize memory
        self.memory = A2CMemory()
        
        # Determine device
        self.device = (
            "mps" if torch.backends.mps.is_available() 
            else "cuda" if torch.cuda.is_available() 
            else "cpu"
        )
        print(f"Using device: {self.device}")
        
        # Training parameters
        self.gamma = 0.99
        self.entropy_coef = 0.01
        
        # Model saving
        self.model_path = "saved_models/"
        os.makedirs(self.model_path, exist_ok=True)
        
        # Metrics and logging
        self.metrics = CuttingStockMetrics()
        self.debug_mode = True
        self.prev_filled_ratio = 0
        self.log_file = open('a2c_reward_log.txt', 'w')
        
        # Episode tracking
        self.steps = 0
        self.initial_products = None
        self.prev_total_products = None
        
        # Reward tracking
        self.last_reward = 0
        self.reward_history = deque(maxlen=10)
        
        # Loss tracking
        self.last_actor_loss = None
        self.last_critic_loss = None

    def calculate_stock_filled_ratio(self, stock):
        """Calculate the filled ratio of a single stock"""
        if stock is None:
            return 0.0
        total_area = stock.shape[0] * stock.shape[1]
        filled_area = np.sum(stock != -1)  # Count non-empty cells
        return filled_area / total_area if total_area > 0 else 0.0

    def calculate_edge_utilization(self, stock):
        """Calculate how well edges are utilized"""
        if stock is None:
            return 0
        edges_used = 0
        height, width = stock.shape
        
        # Check top and bottom edges
        edges_used += np.sum(stock[0, :] != -1)  # Top edge
        edges_used += np.sum(stock[-1, :] != -1)  # Bottom edge
        
        # Check left and right edges
        edges_used += np.sum(stock[:, 0] != -1)  # Left edge
        edges_used += np.sum(stock[:, -1] != -1)  # Right edge
        
        total_edge_cells = 2 * (height + width)
        return edges_used / total_edge_cells if total_edge_cells > 0 else 0

    def calculate_corner_placements(self, stock):
        """Calculate how many corners are utilized"""
        if stock is None:
            return 0
        corners = 0
        height, width = stock.shape
        
        # Check all four corners
        corners += (stock[0, 0] != -1)     # Top-left
        corners += (stock[0, -1] != -1)    # Top-right
        corners += (stock[-1, 0] != -1)    # Bottom-left
        corners += (stock[-1, -1] != -1)   # Bottom-right
        
        return corners

    def update_metrics(self, observation, action, info):
        """Update metrics after each action"""
        if action is not None:
            current_stock = observation['stocks'][action['stock_idx']]
            
            # Update episode metrics
            self.metrics.episode_metrics['filled_ratios'].append(info['filled_ratio'])
            self.metrics.episode_metrics['edge_utilization'].append(
                self.calculate_edge_utilization(current_stock)
            )
            self.metrics.episode_metrics['corner_placements'].append(
                self.calculate_corner_placements(current_stock)
            )
            
            # Update running averages
            self.metrics.running_averages['filled_ratio'].append(info['filled_ratio'])

    def initialize_networks(self, observation):
        """Initialize networks after getting first observation"""
        if self.max_products is None:
            self.max_products = len(observation["products"])
            stock_features = 100 * 3  # max_stocks * features_per_stock
            product_features = self.max_products * 3  # max_products * features_per_product
            global_features = 2
            self.state_dim = stock_features + product_features + global_features
            
            # Initialize networks
            self.actor = ActorNetwork(self.state_dim, self.action_dim).to(self.device)
            self.critic = CriticNetwork(self.state_dim).to(self.device)
            
            # Initialize optimizers
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
            
            # Initialize schedulers
            self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.actor_optimizer, mode='max', factor=0.5, patience=5, verbose=True)
            self.critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.critic_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
            
            # Initialize state normalization
            self.state_mean = torch.zeros(self.state_dim).to(self.device)
            self.state_std = torch.ones(self.state_dim).to(self.device)
    
    def normalize_state(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return (state - self.state_mean) / (self.state_std + 1e-8)
    
    def update_state_normalizer(self, state):
        state = torch.FloatTensor(state).to(self.device)
        self.state_mean = 0.99 * self.state_mean + 0.01 * state.mean()
        self.state_std = 0.99 * self.state_std + 0.01 * state.std()
    
    def get_action(self, observation, info):
        # Initialize networks if first time
        if self.max_products is None:
            self.initialize_networks(observation)
        
        # Khởi tạo initial_products nếu chưa có
        if self.initial_products is None:
            self.initial_products = sum(prod["quantity"] for prod in observation["products"])
        
        state = self.preprocess_observation(observation, info)
        state = self.normalize_state(state).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.actor(state)
            
            # Tăng exploration
            temperature = max(1.5 - (self.steps / 15000), 0.7)  # Tăng temperature và giảm chậm hơn
            logits = logits / temperature
            
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            
            if self.training:
                log_prob = dist.log_prob(action)
                value = self.critic(state)
                self.memory.add(state.cpu(), action.item(), 0, value.item(), log_prob.item(), False)
    
        # Convert to actual placement action
        placement_action = self.convert_action(action.item(), observation)
        
        # If invalid action, try to find a valid one
        if placement_action is None:
            placement_action = self._get_greedy_valid_action(observation)
        
        self.steps += 1
        
        # Add logging after getting action
        if self.debug_mode and self.steps % 100 == 0:
            self._log_step_summary(observation, placement_action, info)
        
        return placement_action
    def _log_step_summary(self, observation, action, info):
        """Log detailed summary of current step"""
        print(f"\n{'='*30} Step {self.steps} Summary {'='*30}\n")
        
        # 1. Efficiency Metrics
        used_stocks = sum(1 for stock in observation['stocks'] if np.any(stock != -1))
        current_total_products = sum(prod["quantity"] for prod in observation["products"])
        products_completed = self.initial_products - current_total_products
        
        print("1. Efficiency Metrics:")
        print(f"  ├── Stocks Used: {used_stocks}")
        print(f"  ├── Products Completed: {products_completed}/{self.initial_products} ({(products_completed/self.initial_products)*100:.1f}%)")
        print(f"  └── Products Remaining: {current_total_products}")
        
        # 2. Current Action Analysis
        print("\n2. Current Action:")
        if action is not None:
            current_stock = observation['stocks'][action['stock_idx']]
            stock_ratio = self.calculate_stock_filled_ratio(current_stock)
            
            print(f"  ├── Stock Index: {action['stock_idx']}")
            print(f"  ├── Position: {action['position']}")
            print(f"  ├── Product Size: {action['size']}")
            print(f"  ├── Current Stock Fill Ratio: {stock_ratio:.3f}")
            
            # Evaluate action quality
            if stock_ratio > 0.8:
                print("  └── Quality: EXCELLENT ⭐⭐⭐ (Stock utilization > 80%)")
            elif stock_ratio > 0.6:
                print("  └── Quality: GOOD ⭐⭐ (Stock utilization > 60%)")
            elif stock_ratio > 0.4:
                print("  └── Quality: FAIR ⭐ (Stock utilization > 40%)")
            else:
                print("  └── Quality: POOR ⚠️ (Low stock utilization)")
        else:
            print("  └── No valid action found ❌")
        
        # 3. Training Progress
        print("\n3. Training Progress:")
        print(f"  ├── Current Reward: {self.last_reward:.3f}")
        avg_reward = np.mean(list(self.reward_history)) if self.reward_history else 0
        print(f"  ├── Average Reward (last 10): {avg_reward:.3f}")
        if hasattr(self, 'last_actor_loss') and hasattr(self, 'last_critic_loss'):
            print(f"  ├── Actor Loss: {self.last_actor_loss:.3f}" if self.last_actor_loss is not None else "  ├── Actor Loss: N/A")
            print(f"  └── Critic Loss: {self.last_critic_loss:.3f}" if self.last_critic_loss is not None else "  └── Critic Loss: N/A")
    def preprocess_observation(self, observation, info):
        """Convert observation to state tensor with fixed dimensions"""
        stocks = observation["stocks"]
        products = observation["products"]
        
        # Stock features - đảm bảo đủ 100 stocks
        stock_features = []
        for stock in stocks[:100]:  # Lấy tối đa 100 stocks
            stock_w, stock_h = self._get_stock_size_(stock)
            used_space = np.sum(stock != -1)
            total_space = stock_w * stock_h
            stock_features.extend([
                stock_w / 10.0,  # Normalized width
                stock_h / 10.0,  # Normalized height
                used_space / total_space  # Utilization ratio
            ])
        
        # Pad nếu thiếu stocks
        while len(stock_features) < 300:  # 100 stocks * 3 features
            stock_features.extend([0, 0, 0])
        
        # Product features - đảm bảo đủ 18 products
        prod_features = []
        for prod in products[:self.max_products]:  # Lấy tối đa số products từ môi trường
            if prod["quantity"] > 0:
                prod_features.extend([
                    prod["size"][0] / 10.0,  # Normalized width
                    prod["size"][1] / 10.0,  # Normalized height
                    min(prod["quantity"], 10) / 10.0  # Normalized quantity
                ])
        
        # Pad nếu thiếu products
        while len(prod_features) < self.max_products * 3:  # max_products * 3 features
            prod_features.extend([0, 0, 0])
        
        global_features = [
            info.get('filled_ratio', 0),
            self.steps / 1000.0  # Normalized step count
        ]
        
        # Combine all features
        state = np.array(stock_features + prod_features + global_features, dtype=np.float32)
        return torch.FloatTensor(state)
    
    def convert_action(self, action_idx, observation):
        # Convert network output to placement parameters
        """
        Convert an action index into a valid product placement within the given observation.

        Args:
            action_idx (int): The index of the action chosen by the policy network.
            observation (dict): A dictionary containing the current state.

        Returns:
            dict: A dictionary with keys 'stock_idx', 'size', and 'position'.
        """
        if action_idx is None:
            return None

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
    
    def compute_advantages(self, rewards, values, dones):
        """Compute advantages using TD(0)"""
        advantages = []
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages.insert(0, delta)  # TD(0) advantage
        return torch.tensor(advantages, dtype=torch.float32).to(self.device)
    
    def update_policy(self, reward=None, done=None, info=None):
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

        # Only perform A2C update when we have enough experience
        if len(self.memory.states) >= 128:  # Adjust batch size as needed
            self._update_networks()
            self.memory.clear()
            
    def _update_networks(self):
        """Internal method to perform the actual A2C update"""
        if not self.training or len(self.memory.states) == 0:
            return
            
        # Convert memory to tensors
        states = torch.stack(self.memory.states).to(self.device)
        actions = torch.tensor(self.memory.actions).to(self.device)
        rewards = torch.tensor(self.memory.rewards, dtype=torch.float32).to(self.device)
        old_values = torch.tensor(self.memory.values, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(self.memory.log_probs, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.memory.dones, dtype=torch.float32).to(self.device)
        
        # Calculate advantages and returns using TD(0)
        advantages = self.compute_advantages(rewards, old_values, dones)
        returns = advantages + old_values  # In A2C, returns = advantage + value
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        print(f"\nUpdating networks with {len(self.memory.states)} samples...")
        
        # Compute current policy distributions
        logits = self.actor(states)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Calculate actor loss (policy gradient)
        actor_loss = -(new_log_probs * advantages).mean()
        
        # Normalize returns before computing loss
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Critic prediction
        value_pred = self.critic(states).squeeze(-1)
        
        # Normalized critic loss
        critic_loss = F.mse_loss(value_pred, returns)
        
        # Store the losses
        self.last_actor_loss = actor_loss.item()
        self.last_critic_loss = critic_loss.item()
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
        print(f"Losses - Actor: {actor_loss.item():.3f}, Critic: {critic_loss.item():.3f}")
        
        # Update networks
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        # Update learning rates if using schedulers
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
        """Calculate comprehensive reward similar to PPO"""
        if action is None:
            return -2.0
            
        reward = 0
        current_filled_ratio = info.get('filled_ratio', 0)
        filled_ratio_change = current_filled_ratio - self.prev_filled_ratio
        
        # 1. Filled Ratio Reward (30%)
        filled_ratio_reward = filled_ratio_change * 15.0
        reward += filled_ratio_reward
        
        # 2. Space Utilization (30%)
        if action is not None:
            stock = observation["stocks"][action["stock_idx"]]
            pos_x, pos_y = action["position"]
            size_w, size_h = action["size"]
            
            # Calculate stock utilization
            stock_w, stock_h = self._get_stock_size_(stock)
            stock_filled_ratio = self.calculate_stock_filled_ratio(stock)
            reward += stock_filled_ratio * 10.0
            
            # Edge and corner bonuses
            if pos_x == 0 or pos_x + size_w == stock_w:
                reward += 0.2
            if pos_y == 0 or pos_y + size_h == stock_h:
                reward += 0.2
            if (pos_x == 0 and pos_y == 0) or \
               (pos_x == 0 and pos_y + size_h == stock_h) or \
               (pos_x + size_w == stock_w and pos_y == 0) or \
               (pos_x + size_w == stock_w and pos_y + size_h == stock_h):
                reward += 0.3
        
        # 3. Product Completion Reward
        total_remaining = sum(prod["quantity"] for prod in observation["products"])
        if self.prev_total_products is None:
            self.prev_total_products = total_remaining
        products_completed = self.prev_total_products - total_remaining
        
        if products_completed > 0:
            used_stocks = sum(1 for stock in observation['stocks'] if np.any(stock != -1))
            completion_bonus = products_completed * (5.0 / max(1, used_stocks))
            reward += completion_bonus
            
            # Extra bonus for completing products when few remain
            if total_remaining <= 3:
                reward += 2.0
        
        # 4. Stock Usage Penalty
        used_stocks = sum(1 for stock in observation['stocks'] if np.any(stock != -1))
        stock_penalty = -0.5 * used_stocks
        reward += stock_penalty
        
        # 5. Waste Penalty
        if action is not None:
            waste_area = (stock_w * stock_h) - (size_w * size_h)
            waste_penalty = -0.05 * (waste_area / (stock_w * stock_h))
            reward += waste_penalty
        
        # Update tracking variables
        self.prev_filled_ratio = current_filled_ratio
        self.prev_total_products = total_remaining
        
        # Update reward history
        self.last_reward = reward
        self.reward_history.append(reward)
        
        return reward
    def calculate_stock_filled_ratio(self, stock):
        """Calculate filled ratio for a single stock"""
        if stock is None:
            return 0.0
        stock_w, stock_h = self._get_stock_size_(stock)
        total_area = stock_w * stock_h
        used_area = np.sum(stock != -1)
        return used_area / total_area
    def calculate_space_utilization(self, stock, pos_x, pos_y, size_w, size_h):
        """Calculate how efficiently the remaining space can be used"""
        stock_w, stock_h = self._get_stock_size_(stock)
        
        # Calculate remaining rectangles after placement
        remaining_areas = []
        
        # Right rectangle
        if pos_x + size_w < stock_w:
            right_w = stock_w - (pos_x + size_w)
            right_h = stock_h
            remaining_areas.append(right_w * right_h)
        
        # Top rectangle
        if pos_y + size_h < stock_h:
            top_w = size_w
            top_h = stock_h - (pos_y + size_h)
            remaining_areas.append(top_w * top_h)
        
        # Calculate utilization score
        total_remaining = sum(remaining_areas)
        stock_area = stock_w * stock_h
        used_area = size_w * size_h
        
        # Reward based on how much usable space remains
        space_efficiency = used_area / stock_area
        remaining_ratio = total_remaining / stock_area
        
        # Return weighted score
        return space_efficiency * 3.0 + remaining_ratio * 2.0

    def calculate_stock_penalty(self, observation):
        used_stocks = sum(1 for stock in observation['stocks'] if np.any(stock > 0))
        stock_penalty = -0.2 * used_stocks
        return stock_penalty

    def _is_good_pattern(self, pos_x, pos_y, size_w, size_h, stock):
        """Evaluate if the placement creates a good cutting pattern"""
        stock_w, stock_h = self._get_stock_size_(stock)
        
        # Check edge alignment
        is_edge_aligned = (pos_x == 0 or pos_x + size_w == stock_w or
                          pos_y == 0 or pos_y + size_h == stock_h)
        
        # Calculate remaining usable space
        used_area = size_w * size_h
        total_area = stock_w * stock_h
        remaining_ratio = 1 - (used_area / total_area)
        
        # Check if the remaining space is still usable
        min_dimension = min(stock_w, stock_h)
        has_usable_space = (remaining_ratio >= 0.3 and  # At least 30% space left
                           min_dimension >= min(size_w, size_h))  # Can fit similar pieces
        
        return is_edge_aligned and has_usable_space
    
    def _get_random_valid_action(self, observation):
        """Get a random valid action when the policy's chosen action is invalid."""
        for stock_idx, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            
            for prod in observation["products"]:
                if prod["quantity"] > 0:
                    prod_w, prod_h = prod["size"]
                    
                    # Try random positions until a valid one is found
                    for _ in range(10):  # Limit attempts to avoid infinite loop
                        pos_x = np.random.randint(0, max(1, stock_w - prod_w + 1))
                        pos_y = np.random.randint(0, max(1, stock_h - prod_h + 1))
                        
                        if self._can_place_(stock, (pos_x, pos_y), prod["size"]):
                            return {
                                "stock_idx": stock_idx,
                                "size": prod["size"],
                                "position": (pos_x, pos_y)
                            }
        
        return None  # Return None if no valid action is found
    
    def evaluate_cutting_pattern(self, observation, action, info):
        """Evaluate the quality of a cutting pattern"""
        if action is None:
            return None
        
        stock = observation['stocks'][action['stock_idx']]
        stock_w, stock_h = self._get_stock_size_(stock)
        pos_x, pos_y = action['position']
        size_w, size_h = action['size']
        
        # Calculate metrics
        edge_contact = 0
        if pos_x == 0 or pos_x + size_w == stock_w:
            edge_contact += 1
        if pos_y == 0 or pos_y + size_h == stock_h:
            edge_contact += 1
        
        is_corner = (pos_x == 0 or pos_x + size_w == stock_w) and \
                   (pos_y == 0 or pos_y + size_h == stock_h)
        
        self.metrics.episode_metrics['edge_utilization'].append(edge_contact)
        self.metrics.episode_metrics['corner_placements'].append(int(is_corner))
        
        return {
            'edge_contact': edge_contact,
            'is_corner': is_corner,
            'piece_area': size_w * size_h,
            'position_quality': pos_x == 0 or pos_y == 0  # Preference for edge placement
        }
    
    def plot_training_progress(self):
        """Plot training metrics"""
        plt.figure(figsize=(15, 10))
        
        # Plot rewards
        plt.subplot(221)
        plt.plot(self.metrics.episode_metrics['rewards'])
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot filled ratios
        plt.subplot(222)
        plt.plot(self.metrics.episode_metrics['filled_ratios'])
        plt.title('Filled Ratios')
        plt.xlabel('Episode')
        plt.ylabel('Ratio')
        
        # Plot edge utilization
        plt.subplot(223)
        plt.plot(self.metrics.episode_metrics['edge_utilization'])
        plt.title('Edge Utilization')
        plt.xlabel('Episode')
        plt.ylabel('Count')
        
        # Plot corner placements
        plt.subplot(224)
        plt.plot(self.metrics.episode_metrics['corner_placements'])
        plt.title('Corner Placements')
        plt.xlabel('Episode')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'training_progress.png'))
        plt.close()
    
    # ===========================
    # Helper Methods (Placeholders)
    # ===========================
    def _get_stock_size_(self, stock):
        """
        Placeholder method to extract stock size.
        Replace with actual logic based on your environment's stock representation.
        """
        # Assuming stock is represented as a 2D NumPy array where -1 indicates unused space
        # and positive integers indicate used space
        return stock.shape[1], stock.shape[0]  # width, height

    def _can_place_(self, stock, position, size):
        """
        Placeholder method to check if a product can be placed on the stock.
        Replace with actual logic based on your environment's placement rules.
        """
        pos_x, pos_y = position
        size_w, size_h = size
        stock_w, stock_h = self._get_stock_size_(stock)
        
        # Check boundaries
        if pos_x + size_w > stock_w or pos_y + size_h > stock_h:
            return False
        
        # Check if the space is free (assuming -1 indicates free space)
        return np.all(stock[pos_y:pos_y+size_h, pos_x:pos_x+size_w] == -1)
    
    def _find_unusable_gaps(self, stock, min_size):
        """Find number of small gaps that are too small for any remaining product"""
        gaps = 0
        # Implement gap detection logic here
        # Example: Check for isolated empty spaces smaller than min_size x min_size
        return gaps
    
    def _get_greedy_valid_action(self, observation):
        """Find a valid action prioritizing larger products"""
        # Sort products by area and remaining quantity
        products = [(i, prod) for i, prod in enumerate(observation["products"]) if prod["quantity"] > 0]
        products.sort(key=lambda x: (
            x[1]["size"][0] * x[1]["size"][1] * x[1]["quantity"],  # Ưu tiên products có tổng diện tích lớn
            x[1]["quantity"]  # Thứ yếu là số lượng còn lại
        ), reverse=True)
        
        # Try each stock
        for stock_idx, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            
            # Try each product
            for _, prod in products:
                size_w, size_h = prod["size"]
                
                # Try all valid positions
                for pos_x in range(0, stock_w - size_w + 1):
                    for pos_y in range(0, stock_h - size_h + 1):
                        if self._can_place_(stock, (pos_x, pos_y), prod["size"]):
                            return {
                                "stock_idx": stock_idx,
                                "size": prod["size"],
                                "position": (pos_x, pos_y)
                            }
        
        return None
    
    def log_episode_summary(self, steps, filled_ratio, episode_reward, observation):
        """Log detailed episode summary with performance analysis"""
        used_stocks = sum(1 for stock in observation['stocks'] if np.any(stock != -1))
        current_total_products = sum(prod["quantity"] for prod in observation["products"])
        products_completed = self.initial_products - current_total_products
        
        print(f"\n{'='*30} Episode Summary {'='*30}")
        
        # 1. Basic Metrics
        print("\n1. Basic Metrics:")
        print(f"  ├── Total Steps: {steps}")
        print(f"  ├── Final Filled Ratio: {filled_ratio:.3f}")
        print(f"  └── Episode Reward: {episode_reward:.2f}")
        
        # 2. Efficiency Analysis
        print("\n2. Efficiency Analysis:")
        products_per_stock = products_completed / max(1, used_stocks)
        efficiency_score = (products_completed / self.initial_products) / max(1, used_stocks)
        
        print(f"  ├── Stocks Used: {used_stocks}")
        print(f"  ├── Products Completed: {products_completed}/{self.initial_products} ({(products_completed/self.initial_products)*100:.1f}%)")
        print(f"  ├── Products per Stock: {products_per_stock:.2f}")
        print(f"  └── Efficiency Score: {efficiency_score:.3f}")
        
        # 3. Performance Rating
        print("\n3. Performance Rating:")
        filled_rating = "⭐⭐⭐" if filled_ratio > 0.8 else "⭐⭐" if filled_ratio > 0.6 else "⭐"
        efficiency_rating = "⭐⭐⭐" if efficiency_score > 0.3 else "⭐⭐" if efficiency_score > 0.2 else "⭐"
        products_rating = "⭐⭐⭐" if products_completed/self.initial_products > 0.9 else "⭐⭐" if products_completed/self.initial_products > 0.7 else "⭐"
        
        print(f"  ├── Fill Quality: {filled_rating}")
        print(f"  ├── Efficiency: {efficiency_rating}")
        print(f"  └── Completion: {products_rating}")
        
        # 4. Analysis & Recommendations
        print("\n4. Analysis & Recommendations:")
        recommendations = []
        
        if used_stocks > products_completed/2:
            recommendations.append("⚠️ High stock usage - Consider optimizing stock utilization")
        if filled_ratio < 0.6:
            recommendations.append("⚠️ Low fill ratio - Need better placement strategy")
        if products_per_stock < 1.5:
            recommendations.append("⚠️ Low products per stock - Try to pack more products in each stock")
        if products_completed/self.initial_products < 0.8:
            recommendations.append("⚠️ Low completion rate - Focus on completing more products")
        
        if recommendations:
            for rec in recommendations:
                print(f"  ├── {rec}")
        else:
            print("  └── Good performance! No major improvements needed ✅")
        
        # 5. Overall Grade
        print("\n5. Overall Grade:")
        avg_score = (filled_ratio + efficiency_score + products_completed/self.initial_products) / 3
        if avg_score > 0.8:
            grade = "A 🏆 (Excellent)"
        elif avg_score > 0.6:
            grade = "B 🎯 (Good)"
        elif avg_score > 0.4:
            grade = "C 📈 (Fair)"
        else:
            grade = "D ⚠️ (Needs Improvement)"
        
        print(f"  └── Final Grade: {grade} (Score: {avg_score:.3f})")
        print("="*70 + "\n")
    
