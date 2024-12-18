import torch
import numpy as np
import gymnasium as gym
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
    
# Define the value network (baseline)
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)

# Function to select an action based on policy probabilities
def select_action(policy, state, test=False):
    state = torch.FloatTensor(state)
    action_probs = policy(state)
    if test:
        action = np.argmax(action_probs.detach().numpy())
    else:
        action = np.random.choice(len(action_probs), p=action_probs.detach().numpy())
    return action

# Function to compute the return (discounted reward)
def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

# Hyperparameters
policy_learning_rate = 0.001
value_learning_rate = 0.001
gamma = 0.95  # Discount factor
num_training_episodes = 500
num_test_episodes = 10
test_interval = 10

# Train and test the agent
def train_and_test():
    env = gym.make("CartPole-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim)
    value = ValueNetwork(state_dim)

    optimizer_policy = optim.AdamW(policy.parameters(), lr=policy_learning_rate)
    optimizer_value = optim.AdamW(value.parameters(), lr=value_learning_rate)

    training_rewards = []
    test_rewards = []

    for episode in range(1, num_training_episodes + 1):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        values = []

        # Generate an episode
        while True:
            state_tensor = torch.FloatTensor(state)
            action = select_action(policy, state)
            log_prob = torch.log(policy(state_tensor)[action])
            value_estimate = value(state_tensor)

            next_state, reward, terminated, truncated, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value_estimate)

            state = next_state
            if terminated or truncated:
                break

        # Compute discounted returns
        returns = compute_returns(rewards, gamma)
        returns_tensor = torch.FloatTensor(returns)

        # Update the value network (MSE loss between predicted value and return)
        values_tensor = torch.cat(values).squeeze()
        value_loss = nn.MSELoss()(values_tensor, returns_tensor)

        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

        # Compute advantages (A_t = G_t - V(s_t))
        advantages = returns_tensor - values_tensor.detach()

        # Update the policy network (REINFORCE with advantage)
        log_probs_tensor = torch.stack(log_probs)
        policy_loss = -(log_probs_tensor * advantages).sum()

        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        # Track training rewards
        training_rewards.append(sum(rewards))

        # Test the agent every few episodes
        if episode % test_interval == 0:
            avg_test_reward = test_agent(policy, env, num_test_episodes)
            test_rewards.append(avg_test_reward)
            print(f"Episode {episode}, Test Reward: {avg_test_reward}")

    env.close()

    # Plot learning curve
    plot_learning_curve(test_rewards, test_interval)

# Function to test the agent and compute average reward
def test_agent(policy, env, num_episodes):
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        while True:
            action = select_action(policy, state, test=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

# Function to plot the learning curve
def plot_learning_curve(test_rewards, test_interval):
    plt.figure(figsize=(10,6))
    plt.plot(range(test_interval, len(test_rewards) * test_interval + 1, test_interval), test_rewards)
    plt.xlabel("Training Episodes")
    plt.ylabel("Test Reward")
    plt.title("Learning Curve: Test Reward vs Training Episodes")
    plt.grid()
    plt.savefig('cart_pole_testgraph.png',dpi=400)

if __name__ == "__main__":
    train_and_test()