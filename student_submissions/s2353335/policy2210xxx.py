from policy import Policy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import os


class SmartGreedy(Policy):
    def __init__(self):
        pass
        
    def get_action(self, observation, info):
        """Implements a smarter greedy approach"""
        list_prods = observation["products"]
        stocks = observation["stocks"]
        
        best_action = None
        best_score = float('-inf')
        
        # For each product that has remaining quantity
        for prod in list_prods:
            if prod["quantity"] <= 0:
                continue
                
            prod_size = prod["size"]
            prod_w, prod_h = prod_size
            
            # Try each stock
            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)
                
                # Skip if product doesn't fit
                if stock_w < prod_w or stock_h < prod_h:
                    continue
                    
                # Try each possible position
                for pos_x in range(stock_w - prod_w + 1):
                    for pos_y in range(stock_h - prod_h + 1):
                        if not self._can_place_(stock, (pos_x, pos_y), prod_size):
                            continue
                            
                        # Calculate score based on position
                        score = self._calculate_placement_score(
                            pos_x, pos_y,
                            prod_w, prod_h,
                            stock_w, stock_h
                        )
                        
                        # Update best action if this is better
                        if score > best_score:
                            best_score = score
                            best_action = {
                                "stock_idx": stock_idx,
                                "size": prod_size,
                                "position": (pos_x, pos_y)
                            }
        
        # If no valid action found, fall back to random valid action
        if best_action is None:
            return self._get_random_valid_action(observation)
            
        return best_action
    
    def _calculate_placement_score(self, pos_x, pos_y, prod_w, prod_h, stock_w, stock_h):
        """Calculate a score for placing a product at a given position"""
        score = 0
        
        # 1. Prefer corners
        if (pos_x == 0 or pos_x + prod_w == stock_w) and \
           (pos_y == 0 or pos_y + prod_h == stock_h):
            score += 5
            
        # 2. Prefer edges
        elif pos_x == 0 or pos_x + prod_w == stock_w or \
             pos_y == 0 or pos_y + prod_h == stock_h:
            score += 3
            
        # 3. Penalize central placements
        center_x = abs(pos_x + prod_w/2 - stock_w/2)
        center_y = abs(pos_y + prod_h/2 - stock_h/2)
        score -= (center_x + center_y) * 0.1
        
        return score
    
    def _get_random_valid_action(self, observation):
        """Fallback method to get a random valid action"""
        list_prods = observation["products"]
        
        for prod in list_prods:
            if prod["quantity"] <= 0:
                continue
                
            prod_size = prod["size"]
            
            # Try each stock randomly
            stock_indices = list(range(len(observation["stocks"])))
            np.random.shuffle(stock_indices)
            
            for stock_idx in stock_indices:
                stock = observation["stocks"][stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)
                
                if stock_w < prod_size[0] or stock_h < prod_size[1]:
                    continue
                    
                # Try random positions
                for _ in range(10):  # Limit attempts
                    pos_x = np.random.randint(0, stock_w - prod_size[0] + 1)
                    pos_y = np.random.randint(0, stock_h - prod_size[1] + 1)
                    
                    if self._can_place_(stock, (pos_x, pos_y), prod_size):
                        return {
                            "stock_idx": stock_idx,
                            "size": prod_size,
                            "position": (pos_x, pos_y)
                        }
        
        # If still no valid action found, return a default action
        return {
            "stock_idx": 0,
            "size": [1, 1],
            "position": (0, 0)
        }

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, state):
        # Add batch dimension if necessary
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        logits = self.fc3(x)
        
        # Remove batch dimension if it was added
        if logits.size(0) == 1:
            logits = logits.squeeze(0)
            
        return F.softmax(logits, dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, state):
        # Add batch dimension if necessary
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        value = self.fc3(x)
        
        # Remove batch dimension if it was added
        if value.size(0) == 1:
            value = value.squeeze(0)
            
        return value

class ActorCriticPolicy(Policy):
    def __init__(self):
        # Adjust state dimension based on feature vector size
        max_stocks = 10
        max_products = 10
        stock_features = max_stocks * 3  # 3 features per stock
        product_features = max_products * 3  # 3 features per product
        global_features = 2  # filled_ratio and step count
        self.state_dim = stock_features + product_features + global_features
        
        # Action space: stock_idx * positions (assuming 5x5 grid for each stock)
        self.action_dim = max_stocks * 25
        
        # Check MPS availability
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        else:
            self.device = torch.device("cpu")
            print("MPS not available, using CPU")
            
        # Initialize networks and move to device
        self.actor = ActorNetwork(self.state_dim, self.action_dim).to(self.device)
        self.critic = CriticNetwork(self.state_dim).to(self.device)
        
        
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.1)
        # Optimizers
        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(),
            lr=3e-4,
            weight_decay=0.01
        )
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=3e-4,
            weight_decay=0.01
        )
        
        # Thêm learning rate scheduler
        self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode='max', factor=0.5, patience=5
        )
        self.critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Other initializations
        self.gamma = 0.99
        self.entropy_coef = 0.02  # Tăng từ 0.01
        self.training = True
        self.current_episode = []
        self.prev_filled_ratio = 0.0
        
        # Add tracking metrics
        self.episode_metrics = {
            'steps': 0,
            'total_reward': 0,
            'filled_ratios': [],
            'invalid_actions': 0,
            'completed_products': 0
        }
        
        # Thêm cache cho state normalization
        self.state_cache = {}
        self.cache_size = 1000
        
        # Add path for saving models
        self.model_path = "saved_models/"
        os.makedirs(self.model_path, exist_ok=True)
    
    def save_model(self, episode=None):
        """Save the model state"""
        try:
            # Create filename with episode number if provided
            filename = f"model_actor_critics_{episode}" if episode is not None else "model_final_actor_critics"
            
            # Save actor and critic states
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'actor_scheduler_state_dict': self.actor_scheduler.state_dict(),
                'critic_scheduler_state_dict': self.critic_scheduler.state_dict(),
                'episode': episode
            }, os.path.join(self.model_path, f"{filename}.pt"))
            
            print(f"Model saved successfully to {filename}.pt")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def load_model(self, filename):
        """Load the model state"""
        try:
            checkpoint = torch.load(os.path.join(self.model_path, filename))
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
            self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])
            
            print(f"Model loaded successfully from {filename}")
            return checkpoint.get('episode', None)
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

    def _normalize_state(self, state):
        """Normalize state using running statistics"""
        if self.state_mean is None:
            self.state_mean = torch.zeros_like(state)
            self.state_std = torch.ones_like(state)
        
        # Update running statistics
        if self.training:
            with torch.no_grad():
                self.state_mean = 0.99 * self.state_mean + 0.01 * state
                self.state_std = 0.99 * self.state_std + 0.01 * (state - self.state_mean).pow(2)
        
        # Normalize state
        return (state - self.state_mean) / (torch.sqrt(self.state_std) + 1e-8)

    def get_action(self, observation, info):
        state = self._preprocess_state(observation, info)
        state = state.to(self.device)  # Move to MPS
        
        with torch.no_grad():
            action_probs = self.actor(state)
            action_probs = F.softmax(action_probs, dim=-1)
            # Move to CPU for sampling
            action_probs = action_probs.cpu()
            action = torch.multinomial(action_probs, 1).item()
        
        # Convert action to placement parameters
        max_stocks = len(observation["stocks"])
        stock_idx = min(action // 25, max_stocks - 1)
        position = action % 25
        pos_x = position // 5
        pos_y = position % 5
        
        # Find valid placement
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
        
        # If no valid action found, use fallback strategy
        if valid_action is None:
            valid_action = self._get_random_valid_action(observation)
        
        # Ensure we always return a valid action
        if valid_action is None:
            # Last resort: return first possible action
            for stock_idx, stock in enumerate(observation["stocks"]):
                for prod in observation["products"]:
                    if prod["quantity"] > 0:
                        valid_action = {
                            "stock_idx": stock_idx,
                            "size": prod["size"],
                            "position": (0, 0)
                        }
                        break
                if valid_action is not None:
                    break
        
        # Calculate immediate reward
        immediate_reward = self._calculate_reward(valid_action, observation, info)
        
        # Update metrics and logging
        self.episode_metrics['steps'] += 1
        self.episode_metrics['total_reward'] += immediate_reward
        self.episode_metrics['filled_ratios'].append(info.get('filled_ratio', 0))
        
        # Print progress every 100 steps
        if self.episode_metrics['steps'] % 100 == 0:
            print("\n" + "="*30 + f" Step {self.episode_metrics['steps']} Summary " + "="*30)
            print("\n1. Action Details:")
            print(f"  Stock Index: {valid_action['stock_idx']}")
            print(f"  Position: {valid_action['position']}")
            print(f"  Product Size: {valid_action['size']}")
            print(f"  Filled Ratio: {info['filled_ratio']:.3f}")
            print(f"  Reward: {immediate_reward:.3f}")
            
            print("\n2. Products Remaining:")
            for i, prod in enumerate(observation['products']):
                if prod['quantity'] > 0:
                    print(f"  Product {i}: {prod['size']} x {prod['quantity']}")
            
            if self.training:
                print("\n3. Training Metrics:")
                print(f"  Actor Loss: {getattr(self, 'last_actor_loss', 'N/A')}")
                print(f"  Critic Loss: {getattr(self, 'last_critic_loss', 'N/A')}")
            print("="*80 + "\n")
        
        if self.training:
            self.current_episode.append({
                'state': state.cpu(),
                'action': action,
                'immediate_reward': float(immediate_reward)
            })
        
        self.prev_filled_ratio = info.get('filled_ratio', 0)
        return valid_action

    def _calculate_reward(self, valid_action, observation, info):
        """Calculate comprehensive reward with improved components"""
        if valid_action is None:
            return -1.0  # Tăng penalty cho invalid action
            
        reward = 0
        current_filled_ratio = info.get('filled_ratio', 0)
        
        # 1. Filled Ratio Component (30% trọng số)
        filled_ratio_change = current_filled_ratio - self.prev_filled_ratio
        filled_ratio_reward = filled_ratio_change * 20.0  # Tăng từ 10
        reward += filled_ratio_reward
        
        # 2. Placement Quality Component (25% trọng số)
        stock = observation["stocks"][valid_action["stock_idx"]]
        stock_w, stock_h = self._get_stock_size_(stock)
        pos_x, pos_y = valid_action["position"]
        size_w, size_h = valid_action["size"]
        
        # 2.1 Edge Utilization (ưu tiên đặt sát cạnh)
        edge_bonus = 0
        if pos_x == 0 or pos_x + size_w == stock_w:
            edge_bonus += 0.5  # Tăng từ 0.2
        if pos_y == 0 or pos_y + size_h == stock_h:
            edge_bonus += 0.5
        reward += edge_bonus
        
        # 2.2 Corner Bonus (ưu tiên đặt góc)
        if (pos_x == 0 or pos_x + size_w == stock_w) and \
           (pos_y == 0 or pos_y + size_h == stock_h):
            reward += 1.0  # Tăng từ 0.3
        
        # 3. Area Efficiency Component (20% trọng số)
        product_area = size_w * size_h
        stock_area = stock_w * stock_h
        area_efficiency = product_area / stock_area
        area_reward = area_efficiency * 2.0  # Tăng từ 0.5
        reward += area_reward
        
        # 4. Completion Bonus (15% trọng số)
        for prod in observation["products"]:
            if prod["quantity"] == 1 and np.array_equal(prod["size"], valid_action["size"]):
                reward += 2.0  # Tăng từ 0.5 - khuyến khích hoàn thành sản phẩm
        
        # 5. Strategic Placement (10% trọng số)
        # 5.1 Giảm penalty cho việc đặt ở giữa
        center_x = abs(pos_x + size_w/2 - stock_w/2) / stock_w
        center_y = abs(pos_y + size_h/2 - stock_h/2) / stock_h
        center_penalty = -(center_x + center_y) * 0.1  # Giảm từ 0.2
        reward += center_penalty
        
        # 5.2 Bonus cho việc đặt các sản phẩm lớn trước
        relative_size = (size_w * size_h) / (stock_w * stock_h)
        if relative_size > 0.3:  # Nếu sản phẩm chiếm >30% diện tích
            reward += 0.5
        
        # Add diversity bonus
        
        # Debug logging
        # print(f"\nReward Breakdown:")
        # print(f"1. Filled Ratio Change: {filled_ratio_reward:.3f}")
        # print(f"2. Edge/Corner Bonus: {edge_bonus:.3f}")
        # print(f"3. Area Efficiency: {area_reward:.3f}")
        # print(f"4. Center Penalty: {center_penalty:.3f}")
        # print(f"5. Size Bonus: {0.5 if relative_size > 0.3 else 0:.3f}")
        print(f"Total Reward: {reward:.3f}")
        
        return reward

    def update_policy(self, reward, done):
        if not self.training or not self.current_episode:
            return
        
        # Debug print
        print(f"\nUpdating policy with {len(self.current_episode)} transitions")
        
        try:
            # Move batch data to MPS
            states = torch.stack([t['state'] for t in self.current_episode]).to(self.device)
            actions = torch.tensor([t['action'] for t in self.current_episode]).to(self.device)
            rewards = torch.tensor([t['immediate_reward'] for t in self.current_episode], 
                                 dtype=torch.float32).to(self.device)
            
            # Ensure tensors have correct shape
            if states.dim() == 1:
                states = states.unsqueeze(0)
            if actions.dim() == 0:
                actions = actions.unsqueeze(0)
            if rewards.dim() == 0:
                rewards = rewards.unsqueeze(0)
                
            # print(f"Tensor shapes - States: {states.shape}, Actions: {actions.shape}, Rewards: {rewards.shape}")
            
            # Process batch
            with torch.no_grad():
                values = self.critic(states).squeeze()
                if values.dim() == 0:
                    values = values.unsqueeze(0)
                
                next_values = torch.zeros_like(values)
                if len(values) > 1:  # Only set next_values if we have more than one value
                    next_values[:-1] = values[1:]
                
                # Calculate advantages
                advantages = torch.zeros_like(rewards)
                gae = 0
                for t in reversed(range(len(rewards))):
                    delta = rewards[t] + self.gamma * next_values[t] - values[t]
                    gae = delta + self.gamma * 0.95 * gae
                    advantages[t] = gae
                
                returns = advantages + values
            
            # Normalize advantages
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Update actor
            action_probs = self.actor(states)
            dist = torch.distributions.Categorical(F.softmax(action_probs, dim=-1))
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            actor_loss = -(log_probs * advantages).mean() - self.entropy_coef * entropy
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update critic
            value_pred = self.critic(states).squeeze()
            critic_loss = F.mse_loss(value_pred, returns)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            print(f"Losses - Actor: {actor_loss.item():.3f}, Critic: {critic_loss.item():.3f}")
            
            # Store losses for logging
            self.last_actor_loss = actor_loss.item()
            self.last_critic_loss = critic_loss.item()
            
            # Print training update
            # print("\n" + "="*30 + " Training Update " + "="*30)
            # print(f"Batch Statistics:")
            # print(f"1. Batch Size: {len(self.current_episode)}")
            # print(f"2. Average Reward: {rewards.mean():.3f}")
            # print(f"3. Reward Range: [{rewards.min():.3f}, {rewards.max():.3f}]")
            # print(f"4. Actor Loss: {self.last_actor_loss:.3f}")
            # print(f"5. Critic Loss: {self.last_critic_loss:.3f}")
        except Exception as e:
            print(f"Error in update_policy: {str(e)}")
            print(f"Current episode length: {len(self.current_episode)}")
            print(f"Rewards: {rewards}")
            print(f"Values shape: {values.shape if 'values' in locals() else 'Not created'}")
            raise e

    def _preprocess_state(self, observation, info):
        """Convert observation to state tensor"""
        stocks = observation["stocks"]
        products = observation["products"]
        
        # Extract features from stocks
        stock_features = []
        for stock in stocks:
            stock_w, stock_h = self._get_stock_size_(stock)
            used_space = np.sum(stock != -1)
            total_space = stock_w * stock_h
            stock_features.extend([
                stock_w / 10.0,  # Normalize size
                stock_h / 10.0,
                used_space / total_space  # Used space ratio
            ])
                
        # Extract features from products
        prod_features = []
        for prod in products:
            if prod["quantity"] > 0:  # Only consider available products
                prod_features.extend([
                    prod["size"][0] / 10.0,  # Normalize size
                    prod["size"][1] / 10.0,
                    min(prod["quantity"], 10) / 10.0  # Cap and normalize quantity
                ])
        
        # Ensure fixed length by padding or truncating
        max_stocks = 10  # Maximum number of stocks to consider
        max_products = 10  # Maximum number of products to consider
        
        # Pad or truncate stock features
        stock_features = stock_features[:max_stocks*3]  # 3 features per stock
        if len(stock_features) < max_stocks*3:
            stock_features.extend([0] * (max_stocks*3 - len(stock_features)))
        
        # Pad or truncate product features
        prod_features = prod_features[:max_products*3]  # 3 features per product
        if len(prod_features) < max_products*3:
            prod_features.extend([0] * (max_products*3 - len(prod_features)))
        
        # Add global features
        global_features = [
            info.get('filled_ratio', 0),
            len(self.current_episode) / 100.0  # Normalized step count
        ]
        
        # Combine all features
        state = np.array(stock_features + prod_features + global_features, dtype=np.float32)
        
        return torch.FloatTensor(state).to(self.device)

    def _get_stock_size_(self, stock):
        """Get width and height of a stock"""
        stock_w = np.sum(np.any(stock != -2, axis=0))
        stock_h = np.sum(np.any(stock != -2, axis=1))
        return stock_w, stock_h

    def _can_place_(self, stock, position, size):
        """Check if we can place a product at the given position"""
        pos_x, pos_y = position
        prod_w, prod_h = size
        
        if pos_x < 0 or pos_y < 0 or pos_x + prod_w > stock.shape[1] or pos_y + prod_h > stock.shape[0]:
            return False
            
        # Check if all cells are available (-1)
        return np.all(stock[pos_y:pos_y+prod_h, pos_x:pos_x+prod_w] == -1)

    def _get_random_valid_action(self, observation):
        """Fallback method for random valid action"""
        for prod in observation["products"]:
            if prod["quantity"] <= 0:
                continue
                
            for stock_idx, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)
                
                if stock_w < prod["size"][0] or stock_h < prod["size"][1]:
                    continue
                    
                for _ in range(10):  # Limit attempts
                    pos_x = np.random.randint(0, stock_w - prod["size"][0] + 1)
                    pos_y = np.random.randint(0, stock_h - prod["size"][1] + 1)
                    
                    if self._can_place_(stock, (pos_x, pos_y), prod["size"]):
                        return {
                            "stock_idx": stock_idx,
                            "size": prod["size"],
                            "position": (pos_x, pos_y)
                        }
        
        # If still no valid action found, return a default action
        return {
            "stock_idx": 0,
            "size": [1, 1],
            "position": (0, 0)
        }