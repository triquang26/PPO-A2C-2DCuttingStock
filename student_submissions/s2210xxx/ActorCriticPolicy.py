import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from policy import Policy
import matplotlib.pyplot as plt

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

class ActorCriticPolicy(Policy):
    def __init__(self):
        super().__init__()
        # State and action dimensions
        self.max_stocks = 10  # Maximum number of stocks
        self.max_products = 10  # Maximum number of products
        self.grid_size = 5  # 5x5 grid for each stock
        
        # Calculate state_dim and action_dim
        self.state_dim = (
            self.max_stocks * 3 +  # stock features (w, h, filled_ratio)
            self.max_products * 3 + # product features (w, h, quantity) 
            2  # global features (overall_filled_ratio, remaining_products)
        )
        self.action_dim = self.max_stocks * (self.grid_size * self.grid_size)
        
        # Device setup
        self.device = (
            "mps" if torch.backends.mps.is_available() 
            else "cuda" if torch.cuda.is_available() 
            else "cpu"
        )
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.actor = ActorNetwork(self.state_dim, self.action_dim).to(self.device)
        self.critic = CriticNetwork(self.state_dim).to(self.device)
        
        # Actor-Critic parameters
        self.gamma = 0.99
        self.entropy_coef = 0.01
        
        # Training setup
        self.training = True
        self.steps = 0
        self.prev_filled_ratio = 0.0
        
        # Store trajectory information
        self.rewards = deque()
        self.dones = deque()
        self.log_probs = []
        self.values = []
        self.entropies = []
        
        # Optimizers with higher learning rates than PPO
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-3)
        
        # Model saving
        self.model_path = "saved_models/"
        os.makedirs(self.model_path, exist_ok=True)
        
        # State normalization
        self.state_mean = torch.zeros(self.state_dim).to(self.device)
        self.state_std = torch.ones(self.state_dim).to(self.device)
        
        # Learning rate schedulers
        self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        self.critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Metrics and logging
        self.last_actor_loss = None
        self.last_critic_loss = None
        self.metrics = CuttingStockMetrics()
        self.debug_mode = True
        self.log_file = open('actor_critic_reward_log.txt', 'w')
        
    def get_action(self, observation, info):
        """Get action without performing immediate updates"""
        state = self.preprocess_observation(observation, info)
        state = state.to(self.device)
        
        with torch.set_grad_enabled(False):  # Disable gradient computation for action selection
            logits = self.actor(state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            value = self.critic(state)
        
        # Convert action to environment format
        env_action = self.convert_action(action.item(), observation)
        
        # If the chosen action is invalid (None), select a random valid action
        if env_action is None:
            env_action = self._get_random_valid_action(observation)
        
        # If still no valid action is found, use the No-Op action
        if env_action is None:
            env_action = NO_OP_ACTION  # Ensure NO_OP_ACTION is defined as shown later
        
        # Store current state info for later updates
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)
        
        return env_action
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
        
        # If no valid action is found, return None to trigger No-Op
        return None
    
    def update_policy(self, reward, done):
        """Update Actor and Critic networks based on the collected trajectory"""
        # Store the latest reward and done flag
        self.rewards.append(reward)
        self.dones.append(done)
        
        # If the episode is done, perform the update
        if done:
            # Compute discounted returns
            returns = []
            G = 0
            for r, d in zip(reversed(self.rewards), reversed(self.dones)):
                if d:
                    G = 0  # Reset the return if the episode ended
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            
            # Normalize returns for stability
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Convert lists to tensors
            log_probs = torch.stack(self.log_probs)
            values = torch.stack(self.values).squeeze()
            entropies = torch.stack(self.entropies)
            
            # Compute advantages
            advantages = returns - values
            
            # Compute actor loss
            actor_loss = -(log_probs * advantages.detach()).mean()
            
            # Compute critic loss
            critic_loss = advantages.pow(2).mean()
            
            # Compute entropy loss (for exploration)
            entropy_loss = -self.entropy_coef * entropies.mean()
            
            # Total loss
            total_loss = actor_loss + critic_loss + entropy_loss
            
            # Backpropagation
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            # Update learning rates
            self.actor_scheduler.step(returns.mean().item())
            self.critic_scheduler.step(critic_loss.item())
            
            # Logging
            self.last_actor_loss = actor_loss.item()
            self.last_critic_loss = critic_loss.item()
            
            # Clear the trajectory
            self.rewards.clear()
            self.dones.clear()
            self.log_probs.clear()
            self.values.clear()
            self.entropies.clear()

    def _update_networks(self, reward):
        """Perform immediate Actor-Critic update"""
        if not self.training or self.current_state is None:
            return
            
        # Convert reward to tensor and ensure it requires grad
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        
        # Calculate TD error
        with torch.set_grad_enabled(True):
            next_value = self.critic(self.current_state)
            td_error = reward + self.gamma * next_value.detach() - self.current_value
            
            # Actor loss
            actor_loss = -self.current_log_prob * td_error.detach()
            
            # Add entropy bonus
            dist = torch.distributions.Categorical(logits=self.actor(self.current_state))
            entropy = dist.entropy()
            actor_loss = actor_loss - self.entropy_coef * entropy
            
            # Critic loss (ensure scalar)
            critic_loss = td_error.pow(2).mean()
            
            # Store losses for logging
            self.last_actor_loss = actor_loss.item()
            self.last_critic_loss = critic_loss.item()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            
            # Update learning rates
            self.actor_scheduler.step(reward.item())
            self.critic_scheduler.step(critic_loss.item())

    def calculate_reward(self, action, observation, info):
        """Calculate comprehensive reward"""
        if action is None:
            return -2.0
            
        reward = 0
        current_filled_ratio = info.get('filled_ratio', 0)
        filled_ratio_change = current_filled_ratio - self.prev_filled_ratio
        
        # 1. Filled Ratio (30%)
        filled_ratio_reward = filled_ratio_change * 15.0
        reward += filled_ratio_reward
        
        # 2. Pattern Quality (30%)
        pattern_eval = self.evaluate_cutting_pattern(observation, action, info)
        if pattern_eval:
            reward += pattern_eval['edge_contact'] * 1.0
            reward += pattern_eval['is_corner'] * 2.0
            reward += pattern_eval['position_quality'] * 0.5
        
        # 3. Stock Usage Penalty
        reward += self.calculate_stock_penalty(observation)
        
        # 4. Completion Bonus
        if self._is_product_completed(observation, action):
            reward += 3.0
            remaining_products = sum(prod['quantity'] for prod in observation['products'])
            if remaining_products <= 3:
                reward += 2.0
        
        self.prev_filled_ratio = current_filled_ratio
        return reward

    def _log_step_info(self, action, observation, info, reward):
        """Log detailed step information"""
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
        actor_loss_str = f"{self.last_actor_loss:.6f}" if self.last_actor_loss is not None else "N/A"
        critic_loss_str = f"{self.last_critic_loss:.6f}" if self.last_critic_loss is not None else "N/A"
        print(f"  Actor Loss: {actor_loss_str}")
        print(f"  Critic Loss: {critic_loss_str}")
        print("="*80 + "\n")

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
    def normalize_state(self, state):
        """Normalize state using running mean and std"""
        state = torch.FloatTensor(state).to(self.device)
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def update_state_normalizer(self, state):
        """Update running mean and std of states"""
        state = torch.FloatTensor(state).to(self.device)
        self.state_mean = 0.99 * self.state_mean + 0.01 * state.mean()
        self.state_std = 0.99 * self.state_std + 0.01 * state.std()

    def preprocess_observation(self, observation, info):
        """Convert observation to state tensor với padding cố định"""
        # Stock features
        stock_features = []
        for i in range(self.max_stocks):
            if i < len(observation['stocks']):
                stock = observation['stocks'][i]
                stock_w, stock_h = self._get_stock_size_(stock)
                filled_ratio = np.sum(stock > 0) / (stock_w * stock_h)
                stock_features.extend([stock_w, stock_h, filled_ratio])
            else:
                stock_features.extend([0, 0, 0])  # Padding

        # Product features
        product_features = []
        for i in range(self.max_products):
            if i < len(observation['products']):
                product = observation['products'][i]
                w, h = product['size']
                qty = product['quantity']
                product_features.extend([w, h, qty])
            else:
                product_features.extend([0, 0, 0])  # Padding

        # Global features
        global_features = [
            info.get('filled_ratio', 0),
            len([p for p in observation['products'] if p['quantity'] > 0])
        ]

        # Combine all features
        state = np.array(stock_features + product_features + global_features)
        
        # Normalize state
        if self.training:
            self.update_state_normalizer(state)
        return self.normalize_state(state)

    def evaluate_cutting_pattern(self, observation, action, info):
        """Evaluate quality of cutting pattern"""
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
            'position_quality': pos_x == 0 or pos_y == 0
        }

    def calculate_stock_penalty(self, observation):
        """Calculate penalty for using multiple stocks"""
        used_stocks = sum(1 for stock in observation['stocks'] if np.any(stock > 0))
        return -0.5 * used_stocks

    def _is_product_completed(self, observation, action):
        """Check if action completes a product"""
        for product in observation['products']:
            if product['quantity'] == 1 and np.array_equal(product['size'], action['size']):
                return True
        return False

    def _evaluate_pattern_quality(self, stock, pos_x, pos_y, size_w, size_h):
        """Evaluate the quality of placement pattern"""
        pattern_score = 0
        
        # Edge alignment bonus
        if pos_x == 0 or pos_y == 0:
            pattern_score += 0.5
            
        # Corner placement bonus
        if (pos_x == 0 and pos_y == 0) or \
           (pos_x == 0 and pos_y + size_h == stock.shape[0]) or \
           (pos_x + size_w == stock.shape[1] and pos_y == 0) or \
           (pos_x + size_w == stock.shape[1] and pos_y + size_h == stock.shape[0]):
            pattern_score += 1.0
            
        return pattern_score

    def _get_stock_size_(self, stock):
        """Get width and height of stock"""
        return stock.shape[1], stock.shape[0]

    def convert_action(self, action_idx, observation):
        """Convert network output to environment action"""
        stock_idx = action_idx // (self.grid_size * self.grid_size)
        position_idx = action_idx % (self.grid_size * self.grid_size)
        
        if stock_idx >= len(observation['stocks']):
            return None
            
        stock = observation['stocks'][stock_idx]
        stock_w, stock_h = self._get_stock_size_(stock)
        
        # Convert position index to x,y coordinates
        pos_x = (position_idx % self.grid_size) * (stock_w // self.grid_size)
        pos_y = (position_idx // self.grid_size) * (stock_h // self.grid_size)
        
        # Find best fitting product
        for prod in observation['products']:
            if prod['quantity'] > 0:
                size_w, size_h = prod['size']
                if (pos_x + size_w <= stock_w and 
                    pos_y + size_h <= stock_h and 
                    self._is_position_valid(stock, pos_x, pos_y, size_w, size_h)):
                    return {
                        "stock_idx": stock_idx,
                        "size": prod['size'],
                        "position": (pos_x, pos_y)
                    }
        return None

    def _is_position_valid(self, stock, pos_x, pos_y, size_w, size_h):
        """Check if position is valid for placement"""
        if pos_x + size_w > stock.shape[1] or pos_y + size_h > stock.shape[0]:
            return False
        
        # Check if area is empty
        return not np.any(stock[pos_y:pos_y+size_h, pos_x:pos_x+size_w] > 0)

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

    