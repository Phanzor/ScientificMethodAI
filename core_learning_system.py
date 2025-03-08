import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from time import sleep

# Neural network for state and reward prediction
class StateRewardPredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, state_size=4, reward_size=1):
        super(StateRewardPredictor, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.state_head = nn.Linear(hidden_size, state_size)
        self.reward_head = nn.Linear(hidden_size, reward_size)
    
    def forward(self, x):
        shared_out = self.shared(x)
        next_state = self.state_head(shared_out)
        reward = self.reward_head(shared_out)
        return next_state, reward

# Set up environment and model
env = gym.make("CartPole-v1", render_mode="human")
predictor = StateRewardPredictor()
optimizer = optim.Adam(predictor.parameters(), lr=0.001)
state_criterion = nn.MSELoss()
reward_criterion = nn.MSELoss()

# Hypothesis class with test counter
class Hypothesis:
    def __init__(self, condition, action, expected_effect):
        self.condition = condition
        self.action = action
        self.expected_effect = expected_effect
        self.confidence = 0.5
        self.tests = 0
        self.successes = 0
    
    def test(self, observation, predictor):
        if self.condition(observation):
            pred_state, _ = generate_hypotheses(predictor, observation, self.action)
            self.tests += 1
            return pred_state.numpy()
        return None
    
    def update_confidence(self, actual_effect):
        expected_sign = -1 if "decreases" in self.expected_effect else 1
        if np.sign(actual_effect) == expected_sign:
            self.successes += 1
        self.confidence = self.successes / max(self.tests, 1)  # Avoid division by zero

# Collect observations with curiosity and hypothesis testing
def collect_and_train(env, predictor, data_log, hypotheses, num_steps=100):
    observation, _ = env.reset()
    
    for _ in range(num_steps):
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        actions = [0, 1]
        errors = []
        with torch.no_grad():
            for action in actions:
                act_tensor = torch.tensor([action], dtype=torch.float32)
                input_tensor = torch.cat((obs_tensor, act_tensor))
                pred_state, pred_reward = predictor(input_tensor)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                env.env.state = observation
                state_error = state_criterion(pred_state, torch.tensor(next_obs, dtype=torch.float32))
                reward_error = reward_criterion(pred_reward, torch.tensor([reward], dtype=torch.float32))
                errors.append(state_error.item() + reward_error.item())
                if terminated or truncated:
                    observation, _ = env.reset()
        
        action = actions[np.argmax(errors)]
        
        next_observation, reward, terminated, truncated, _ = env.step(action)
        data_log.append({
            "observation": observation.copy(),
            "action": action,
            "next_observation": next_observation.copy(),
            "reward": reward
        })
        
        for h in hypotheses:
            pred = h.test(observation, predictor)
            if pred is not None:
                actual_effect = next_observation[2] - observation[2]
                h.update_confidence(actual_effect)
        
        observation = next_observation
        env.render()
        sleep(0.02)
        if terminated or truncated:
            observation, _ = env.reset()
    
    for data in data_log:
        obs = torch.tensor(data["observation"], dtype=torch.float32)
        act = torch.tensor([data["action"]], dtype=torch.float32)
        input_tensor = torch.cat((obs, act))
        target_state = torch.tensor(data["next_observation"], dtype=torch.float32)
        target_reward = torch.tensor([data["reward"]], dtype=torch.float32)
        
        pred_state, pred_reward = predictor(input_tensor)
        state_loss = state_criterion(pred_state, target_state)
        reward_loss = reward_criterion(pred_reward, target_reward)
        loss = state_loss + reward_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return data_log

# Generate hypotheses
def generate_hypotheses(predictor, observation, action):
    with torch.no_grad():
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        act_tensor = torch.tensor([action], dtype=torch.float32)
        input_tensor = torch.cat((obs_tensor, act_tensor))
        pred_state, pred_reward = predictor(input_tensor)
        return pred_state, pred_reward

# Test hypotheses
def test_hypotheses(env, predictor, num_tests=10):
    observation, _ = env.reset()
    state_errors = []
    reward_errors = []
    
    for _ in range(num_tests):
        action = np.random.randint(2)
        pred_state, pred_reward = generate_hypotheses(predictor, observation, action)
        
        next_observation, reward, terminated, truncated, _ = env.step(action)
        state_error = np.mean((pred_state.numpy() - next_observation) ** 2)
        reward_error = (pred_reward.numpy()[0] - reward) ** 2
        state_errors.append(state_error)
        reward_errors.append(reward_error)
        
        observation = next_observation
        env.render()
        sleep(0.02)
        if terminated or truncated:
            observation, _ = env.reset()
    
    return np.mean(state_errors), np.mean(reward_errors)

# Learning loop
def learning_loop(env, predictor, hypotheses, num_cycles=5, steps_per_cycle=200, tests_per_cycle=10):
    data_log = []
    state_history = []
    reward_history = []
    
    for cycle in range(num_cycles):
        print(f"\nCycle {cycle + 1}/{num_cycles}")
        
        print("Collecting and training...")
        data_log = collect_and_train(env, predictor, data_log, hypotheses, num_steps=steps_per_cycle)
        
        print("Testing hypotheses...")
        state_error, reward_error = test_hypotheses(env, predictor, num_tests=tests_per_cycle)
        state_history.append(state_error)
        reward_history.append(reward_error)
        print(f"State Prediction Error: {state_error:.4f}, Reward Prediction Error: {reward_error:.4f}")
        
        total_error = state_error + reward_error
        if total_error > 0.05:
            print("High error detected - focusing on exploration.")
        else:
            print("Low error - predictions are stabilizing.")
        
        print("Hypothesis Confidence:")
        for h in hypotheses:
            print(f"  {h.expected_effect} when {h.condition.__name__} (Action {h.action}): {h.confidence:.2f} "
                  f"({h.successes}/{h.tests} tests)")
    
    return state_history, reward_history

# Visualize progress
def plot_progress(state_history, reward_history):
    plt.plot(state_history, label="State Prediction Error")
    plt.plot(reward_history, label="Reward Prediction Error")
    plt.xlabel("Cycle")
    plt.ylabel("Average Error")
    plt.title("Learning Progress Over Cycles")
    plt.legend()
    plt.show()

# Main execution
def main():
    hypotheses = [
        Hypothesis(
            condition=lambda obs: obs[2] > 0,  # Pole tilted right
            action=1,                          # Move right
            expected_effect="pole angle decreases"
        ),
        Hypothesis(
            condition=lambda obs: obs[2] < 0,  # Pole tilted left
            action=0,                          # Move left
            expected_effect="pole angle decreases"
        )
    ]
    
    print("Starting self-teaching AI with hypotheses...")
    state_history, reward_history = learning_loop(env, predictor, hypotheses)
    print("\nFinal State History:", state_history)
    print("Final Reward History:", reward_history)
    plot_progress(state_history, reward_history)
    
    torch.save(predictor.state_dict(), "predictor.pth")
    print("Model saved to 'predictor.pth'")
    
    env.close()

if __name__ == "__main__":
    main()