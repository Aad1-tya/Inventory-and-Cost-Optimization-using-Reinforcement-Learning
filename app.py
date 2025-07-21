import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.ndimage import uniform_filter1d
import time

# --- RL Environment and Agent Code (from your notebook) ---

# Set a seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

@st.cache_data
def load_data():
    """Loads and preprocesses the sales data, caching the result."""
    try:
        # This assumes the CSV is in the same directory as app.py
        sales = pd.read_csv("sales_train_validation.csv")
    except FileNotFoundError:
        st.error("🚨 Error: `sales_train_validation.csv` not found.")
        st.info("Please make sure the sales data CSV file is in the same folder as this app.")
        return None

    demand_columns = [col for col in sales.columns if col.startswith("d_")]
    non_zero_skus = sales[sales[demand_columns].sum(axis=1) > 100]

    def has_nonzero_early_demand(row, window=30):
        return row[demand_columns[:window]].sum() > 0

    non_zero_early = non_zero_skus[non_zero_skus.apply(has_nonzero_early_demand, axis=1)]

    if len(non_zero_early) == 0:
        st.warning("⚠️ No SKU with sufficient early demand found in the dataset.")
        return None

    row = non_zero_early.iloc[0]
    demand_series = row[demand_columns].values.astype(int)
    return demand_series

class InventoryEnv:
    def __init__(self, demand_data, max_inventory=100, max_order=30, holding_cost=1, stockout_cost=10, order_cost=2):
        self.demand_data = demand_data
        self.max_inventory = max_inventory
        self.max_order = max_order
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.order_cost = order_cost
        self.reset()

    def reset(self):
        self.current_day = 7
        self.inventory_level = np.random.randint(10, 20)
        self.history = []
        return self._get_state()

    def _get_state(self):
        demand_t_1 = self.demand_data[self.current_day - 1]
        avg_demand_7 = np.mean(self.demand_data[self.current_day - 7:self.current_day])
        return np.array([
            self.inventory_level / self.max_inventory,
            demand_t_1 / 10.0,
            avg_demand_7 / 10.0
        ], dtype=np.float32)

    def step(self, action):
        order_qty = int(action)
        demand = self.demand_data[self.current_day]
        self.inventory_level += order_qty
        reward = 0
        shortage = 0

        if demand <= self.inventory_level:
            self.inventory_level -= demand
            fulfilled = demand
        else:
            fulfilled = self.inventory_level
            shortage = demand - self.inventory_level
            self.inventory_level = 0

        reward += fulfilled
        reward -= self.holding_cost * self.inventory_level
        reward -= self.stockout_cost * shortage
        reward -= self.order_cost * order_qty

        self.history.append((self.current_day, self.inventory_level, demand, order_qty, shortage))
        self.current_day += 1
        done = self.current_day >= len(self.demand_data) - 1
        next_state = self._get_state() if not done else np.zeros(3, dtype=np.float32)
        return next_state, reward, done, {'shortage': shortage}


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*samples))
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.int64),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32)
        )
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.05
        self.update_target_steps = 100
        self.steps_done = 0
        self.action_dim = action_dim

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        q_values = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze()
        next_q_values = self.target_network(next_state).max(1)[0]
        expected_q_values = reward + self.gamma * next_q_values * (1 - done)
        loss = F.mse_loss(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.update_target_steps == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.steps_done +=1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def run_training(demand_series, episodes, progress_bar):
    """Encapsulates the training loop."""
    env = InventoryEnv(demand_series)
    state_dim = 3
    action_dim = 31
    agent = DQNAgent(state_dim, action_dim)
    rewards_per_episode = []
    
    # Warm-up buffer (no need to show progress for this)
    state = env.reset()
    for _ in range(1000):
        action = np.random.randint(0, action_dim)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()
        
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
        rewards_per_episode.append(total_reward)
        progress_bar.progress((ep + 1) / episodes, text=f"Training Episode {ep + 1}/{episodes}")
        
    return agent, rewards_per_episode

def evaluate_metrics(agent, env, episodes=10, is_baseline=False):
    """Encapsulates the evaluation logic."""
    # ... (code is identical to your notebook) ...
    # This function is long, so it is kept the same as in the notebook.
    total_rewards, stockouts, holdings, orders = [], [], [], []
    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward, total_stockout, total_holding, total_order = 0, 0, 0, 0
        while not done:
            if is_baseline:
                action = 10 if env.inventory_level < 10 else 0
            else:
                action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_stockout += info.get('shortage', 0)
            total_holding += env.inventory_level
            total_order += action
            total_reward += reward
            state = next_state
        total_rewards.append(total_reward)
        stockouts.append(total_stockout)
        holdings.append(total_holding)
        orders.append(total_order)
    return {
        "Avg Reward": np.mean(total_rewards),
        "Avg Stockouts": np.mean(stockouts),
        "Avg Inventory Held": np.mean(holdings),
        "Avg Orders": np.mean(orders)
    }

# --- Streamlit UI ---

st.set_page_config(page_title="RL Inventory Optimizer", layout="wide")
st.title("📦 RL Inventory Optimization Agent")
st.write("""
This app trains a Deep Q-Network (DQN) agent to manage inventory for a product.
The agent learns to place orders to minimize holding and stockout costs based on historical demand data.
""")

# Load data
demand_series = load_data()

if demand_series is not None:
    st.header("1. Training the Agent")
    st.write("Click the button below to start the training process.")

    # Training parameters
    episodes = st.slider("Select number of training episodes:", 50, 500, 100, 10)

    if st.button("🚀 Train Agent", type="primary"):
        progress_bar = st.progress(0, text="Starting Training...")
        
        # Run the training
        trained_agent, rewards = run_training(demand_series, episodes, progress_bar)
        time.sleep(1) # Give a moment for the progress bar to complete
        progress_bar.progress(1.0, text="Training Complete!")

        st.header("2. Training Results")
        
        # Plotting Training Rewards
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(uniform_filter1d(rewards, size=10))
        ax1.set_title("Smoothed Training Reward Over Episodes")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.grid(True)
        st.pyplot(fig1)

        st.header("3. Performance Evaluation")
        env = InventoryEnv(demand_series)

        # Evaluate DQN Agent
        trained_agent.epsilon = 0.0 # Greedy evaluation
        dqn_metrics = evaluate_metrics(trained_agent, env, episodes=10)
        
        # Evaluate Baseline
        baseline_metrics = evaluate_metrics(trained_agent, env, episodes=10, is_baseline=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🤖 DQN Agent Performance")
            st.metric("Average Reward", f"{dqn_metrics['Avg Reward']:.2f}")
            st.metric("Average Stockouts (units)", f"{dqn_metrics['Avg Stockouts']:.2f}")
            st.metric("Average Inventory Held", f"{dqn_metrics['Avg Inventory Held']:.2f}")

        with col2:
            st.subheader("📊 Baseline Policy Performance")
            st.metric("Average Reward", f"{baseline_metrics['Avg Reward']:.2f}", delta=f"{dqn_metrics['Avg Reward'] - baseline_metrics['Avg Reward']:.2f} vs. DQN", delta_color="normal")
            st.metric("Average Stockouts (units)", f"{baseline_metrics['Avg Stockouts']:.2f}", delta=f"{dqn_metrics['Avg Stockouts'] - baseline_metrics['Avg Stockouts']:.2f} vs. DQN", delta_color="inverse")
            st.metric("Average Inventory Held", f"{baseline_metrics['Avg Inventory Held']:.2f}", delta=f"{dqn_metrics['Avg Inventory Held'] - baseline_metrics['Avg Inventory Held']:.2f} vs. DQN")

        st.header("4. Inventory Dynamics (DQN Agent)")
        st.write("This chart shows how the trained agent manages inventory over a sample episode.")
        
        # Run one evaluation episode to get history
        env.reset()
        done = False
        state = env._get_state()
        while not done:
            action = trained_agent.select_action(state)
            next_state, _, done, _ = env.step(action)
            state = next_state
        
        # Plotting Inventory Dynamics
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        history = env.history
        days = [h[0] for h in history]
        inventory = [h[1] for h in history]
        demand = [h[2] for h in history]
        orders = [h[3] for h in history]
        shortages = [h[4] for h in history]
        
        ax2.plot(days, demand, label='Demand', marker='o', linestyle='--')
        ax2.plot(days, inventory, label='Inventory', marker='s')
        ax2.plot(days, orders, label='Orders Placed', marker='^', linestyle='-.')
        ax2.bar(days, shortages, label='Stockouts', color='red', alpha=0.5)
        ax2.set_xlabel("Day")
        ax2.set_ylabel("Units")
        ax2.set_title("Inventory, Demand, and Orders Over Time (DQN Agent)")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)