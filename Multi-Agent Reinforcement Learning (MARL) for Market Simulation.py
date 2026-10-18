# Author: Rayaan tasnim, 13 aged prodigy

import os
import sys
import math
import random
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MARLMarketSimulator")


class MarketEnvironment:
    """
    A vectorized multi-agent double auction market simulation environment built from scratch.
    Simulates Market Makers (MMs) and Speculators interacting through a shared order book.
    """
    def __init__(self, num_market_makers: int = 2, num_speculators: int = 4, max_steps: int = 500):
        try:
            self.num_mm = num_market_makers
            self.num_spec = num_speculators
            self.total_agents = self.num_mm + self.num_spec
            self.max_steps = max_steps
            
            self.current_step = 0
            self.mid_price = 100.0
            self.volatility = 0.01
            
            # Agent states: [cash, inventory, position_pnl]
            self.agent_cash = np.zeros(self.total_agents, dtype=np.float32) + 10000.0
            self.agent_inventory = np.zeros(self.total_agents, dtype=np.float32)
            self.agent_pnl = np.zeros(self.total_agents, dtype=np.float32)
            
            # Market book state: [best_bid, best_ask, spread, order_imbalance]
            self.best_bid = self.mid_price - 0.5
            self.best_ask = self.mid_price + 0.5
            self.spread = 1.0
            self.order_imbalance = 0.0
            
            logger.info("MarketEnvironment initialized successfully with %d total agents.", self.total_agents)
        except Exception as e:
            logger.error("Failed to initialize MarketEnvironment: %s", traceback.format_exc())
            raise e

    def reset(self) -> np.ndarray:
        try:
            self.current_step = 0
            self.mid_price = 100.0
            self.agent_cash.fill(10000.0)
            self.agent_inventory.fill(0.0)
            self.agent_pnl.fill(0.0)
            
            self.best_bid = self.mid_price - 0.5
            self.best_ask = self.mid_price + 0.5
            self.spread = 1.0
            self.order_imbalance = 0.0
            
            return self._get_observations()
        except Exception as e:
            logger.error("Error during environment reset: %s", traceback.format_exc())
            raise e

    def _get_observations(self) -> np.ndarray:
        """
        Constructs the observation vector for each agent.
        Features per agent: [cash_normalized, inventory_normalized, mid_price, spread, order_imbalance]
        """
        obs = np.zeros((self.total_agents, 5), dtype=np.float32)
        for i in range(self.total_agents):
            obs[i, 0] = self.agent_cash[i] / 10000.0
            obs[i, 1] = self.agent_inventory[i] / 10.0
            obs[i, 2] = self.mid_price / 100.0
            obs[i, 3] = self.spread / 5.0
            obs[i, 4] = self.order_imbalance
        return obs

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        """
        Executes one environment step given discrete actions for all agents.
        Actions:
          0: Hold / No-op
          1: Buy / Post Bid
          2: Sell / Post Ask
        """
        try:
            self.current_step += 1
            
            # Exogenous price shock (Gaussian random walk)
            price_shock = np.random.normal(0.0, self.volatility * self.mid_price)
            self.mid_price = max(10.0, self.mid_price + price_shock)
            
            buy_pressure = 0.0
            sell_pressure = 0.0
            
            # Process agent actions
            for i in range(self.total_agents):
                action = actions[i]
                if action == 1:  # Buy intent
                    buy_pressure += 1.0 if i >= self.num_mm else 0.5
                    execution_price = self.best_ask
                    if self.agent_cash[i] >= execution_price:
                        self.agent_cash[i] -= execution_price
                        self.agent_inventory[i] += 1.0
                elif action == 2:  # Sell intent
                    sell_pressure += 1.0 if i >= self.num_mm else 0.5
                    execution_price = self.best_bid
                    if self.agent_inventory[i] > -10.0:  # Allow shorting up to limit
                        self.agent_cash[i] += execution_price
                        self.agent_inventory[i] -= 1.0

            # Update market microstructure variables
            total_pressure = buy_pressure + sell_pressure + 1e-5
            self.order_imbalance = (buy_pressure - sell_pressure) / total_pressure
            
            # Adjust spread and prices based on order imbalance and MM presence
            self.spread = max(0.2, 1.0 - 0.1 * abs(self.order_imbalance))
            self.best_bid = self.mid_price - self.spread / 2.0
            self.best_ask = self.mid_price + self.spread / 2.0
            
            # Compute rewards: Mark-to-Market Portfolio Value minus Inventory Penalty and Volatility Penalty
            rewards = np.zeros(self.total_agents, dtype=np.float32)
            for i in range(self.total_agents):
                portfolio_value = self.agent_cash[i] + self.agent_inventory[i] * self.mid_price
                pnl = portfolio_value - 10000.0
                reward = pnl - self.agent_pnl[i]  # Step PnL delta
                self.agent_pnl[i] = pnl
                
                # Inventory penalty to prevent runaway exposure
                inventory_penalty = 0.05 * (self.agent_inventory[i] ** 2)
                
                if i < self.num_mm:
                    # Market Maker: penalize spread expansion, reward market stability
                    rewards[i] = reward - inventory_penalty - 0.1 * self.spread
                else:
                    # Speculator: capture directional momentum
                    rewards[i] = reward - inventory_penalty
            
            done = self.current_step >= self.max_steps
            next_observations = self._get_observations()
            
            info = {
                "mid_price": self.mid_price,
                "spread": self.spread,
                "order_imbalance": self.order_imbalance
            }
            
            return next_observations, rewards, done, info
            
        except Exception as e:
            logger.error("Error during environment step execution: %s", traceback.format_exc())
            raise e


class QNetwork(nn.Module):
    """Deep Q-Network architecture for independent agent policy estimation."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(QNetwork, self).__init__()
        try:
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        except Exception as e:
            logger.error("Failed to initialize QNetwork: %s", traceback.format_exc())
            raise e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.net(x)
        except Exception as e:
            logger.error("Forward pass failed in QNetwork: %s", traceback.format_exc())
            raise e


class ReplayBuffer:
    """Experience Replay Buffer built from raw NumPy arrays for decentralized updates."""
    def __init__(self, capacity: int = 10000):
        try:
            self.capacity = capacity
            self.buffer = []
            self.position = 0
        except Exception as e:
            logger.error("Failed to initialize ReplayBuffer: %s", traceback.format_exc())
            raise e

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        try:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.capacity
        except Exception as e:
            logger.error("Error pushing to ReplayBuffer: %s", traceback.format_exc())
            raise e

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
            return (
                states.astype(np.float32),
                actions.astype(np.int64),
                rewards.astype(np.float32),
                next_states.astype(np.float32),
                dones.astype(np.float32)
            )
        except Exception as e:
            logger.error("Error sampling from ReplayBuffer: %s", traceback.format_exc())
            raise e

    def __len__(self) -> int:
        return len(self.buffer)


class IDQNAgent:
    """Independent Deep Q-Network Agent with experience replay and target network updates."""
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001, gamma: float = 0.99):
        try:
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.gamma = gamma
            
            self.q_net = QNetwork(state_dim, action_dim)
            self.target_net = QNetwork(state_dim, action_dim)
            self.target_net.load_state_dict(self.q_net.state_dict())
            
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
            self.memory = ReplayBuffer(capacity=5000)
            
            self.epsilon = 1.0
            self.epsilon_decay = 0.995
            self.epsilon_min = 0.05
        except Exception as e:
            logger.error("Failed to initialize IDQNAgent: %s", traceback.format_exc())
            raise e

    def select_action(self, state: np.ndarray) -> int:
        try:
            if random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return int(torch.argmax(q_values).item())
        except Exception as e:
            logger.error("Error selecting action: %s", traceback.format_exc())
            raise e

    def update_policy(self, batch_size: int = 32) -> Optional[float]:
        try:
            if len(self.memory) < batch_size:
                return None
            
            states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
            
            state_tensor = torch.FloatTensor(states)
            action_tensor = torch.LongTensor(actions).unsqueeze(1)
            reward_tensor = torch.FloatTensor(rewards)
            next_state_tensor = torch.FloatTensor(next_states)
            done_tensor = torch.FloatTensor(dones)
            
            current_q = self.q_net(state_tensor).gather(1, action_tensor).squeeze(1)
            
            with torch.no_grad():
                max_next_q = self.target_net(next_state_tensor).max(1)[0]
                target_q = reward_tensor + (1.0 - done_tensor) * self.gamma * max_next_q
                
            loss = nn.MSELoss()(current_q, target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            return float(loss.item())
        except Exception as e:
            logger.error("Error updating IDQN policy: %s", traceback.format_exc())
            return None

    def update_target_network(self):
        try:
            self.target_net.load_state_dict(self.q_net.state_dict())
        except Exception as e:
            logger.error("Error updating target network: %s", traceback.format_exc())
            raise e

    def decay_epsilon(self):
        try:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        except Exception as e:
            logger.error("Error decaying epsilon: %s", traceback.format_exc())
            raise e


if __name__ == "__main__":
    try:
        logger.info("Starting Multi-Agent Reinforcement Learning Market Pipeline...")
        
        env = MarketEnvironment(num_market_makers=2, num_speculators=2, max_steps=200)
        state_dim = 5
        action_dim = 3  # 0: Hold, 1: Buy, 2: Sell
        
        agents = [IDQNAgent(state_dim, action_dim) for _ in range(env.total_agents)]
        
        num_episodes = 50
        batch_size = 32
        
        for episode in range(num_episodes):
            states = env.reset()
            episode_rewards = np.zeros(env.total_agents)
            done = False
            
            while not done:
                actions = np.array([agents[i].select_action(states[i]) for i in range(env.total_agents)])
                next_states, rewards, done, info = env.step(actions)
                
                for i in range(env.total_agents):
                    agents[i].memory.push(states[i], actions[i], rewards[i], next_states[i], float(done))
                    loss = agents[i].update_policy(batch_size)
                    episode_rewards[i] += rewards[i]
                
                states = next_states
            
            for i in range(env.total_agents):
                agents[i].update_target_network()
                agents[i].decay_epsilon()
                
            if (episode + 1) % 10 == 0:
                logger.info("Episode [%d/%d] Completed | Average Rewards: %s | Mid Price: %.2f | Spread: %.2f",
                            episode + 1, num_episodes, np.round(episode_rewards, 2), info["mid_price"], info["spread"])
                
        logger.info("Training complete. Executing live evaluation episode...")
        
        eval_states = env.reset()
        eval_done = False
        step_count = 0
        
        print("\n--- Evaluation Episode Trace ---")
        while not eval_done and step_count < 20:
            # Greedy actions for evaluation
            eval_actions = np.array([int(torch.argmax(agents[i].q_net(torch.FloatTensor(eval_states[i]).unsqueeze(0))).item()) for i in range(env.total_agents)])
            eval_states, eval_rewards, eval_done, eval_info = env.step(eval_actions)
            step_count += 1
            print(f"Step {step_count:02d} | Actions: {eval_actions} | Mid Price: {eval_info['mid_price']:.2f} | Spread: {eval_info['spread']:.2f} | Imbalance: {eval_info['order_imbalance']:.2f}")
            
        logger.info("Pipeline execution successfully finished.")
        
    except Exception as e:
        logger.critical("Critical error encountered in main execution loop: %s", traceback.format_exc())
        sys.exit(1)