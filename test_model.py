import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from time import sleep

# Define the neural network (must match the saved model's structure)
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

# Load the saved model
def load_model(filepath="predictor.pth"):
    predictor = StateRewardPredictor()
    predictor.load_state_dict(torch.load(filepath))
    predictor.eval()
    return predictor

# Predict next state and reward
def predict(predictor, observation, action):
    with torch.no_grad():
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        act_tensor = torch.tensor([action], dtype=torch.float32)
        input_tensor = torch.cat((obs_tensor, act_tensor))
        pred_state, pred_reward = predictor(input_tensor)
        return pred_state.numpy(), pred_reward.numpy()[0]

# Decide action by minimizing predicted pole angle
def decide_action(predictor, observation):
    obs_tensor = torch.tensor(observation, dtype=torch.float32)
    actions = [0, 1]
    best_action = 0
    min_angle = float('inf')
    
    with torch.no_grad():
        for action in actions:
            act_tensor = torch.tensor([action], dtype=torch.float32)
            input_tensor = torch.cat((obs_tensor, act_tensor))
            pred_state, _ = predictor(input_tensor)
            pred_angle = abs(pred_state[2])  # Pole angle
            if pred_angle < min_angle:
                min_angle = pred_angle
                best_action = action
    return best_action

# Test the model continuously
def test_model(env, predictor):
    episode = 0
    
    while True:  # Run indefinitely until Ctrl+C
        episode += 1
        observation, _ = env.reset()
        total_reward = 0
        print(f"\nStarting Episode {episode}")
        
        while True:
            action = decide_action(predictor, observation)
            pred_state, pred_reward = predict(predictor, observation, action)
            
            next_observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            # Print step details
            print(f"Step: Action={action}, Reward={reward:.4f}, Pred Reward={pred_reward:.4f}")
            print(f"  Actual State: {next_observation}")
            print(f"  Predicted State: {pred_state}")
            
            observation = next_observation
            env.render()
            sleep(0.05)  # Slower pace for visibility (adjust as needed)
            
            if terminated or truncated:
                print(f"Episode {episode} ended. Steps: {int(total_reward)}, Total Reward: {total_reward}")
                sleep(1)  # Pause between episodes
                break
        
        # Optional: Small delay between episodes
        sleep(0.5)

# Main execution
def main():
    env = gym.make("CartPole-v1", render_mode="human")
    predictor = load_model("predictor.pth")
    print("Loaded model from 'predictor.pth'. Running continuous CartPole simulation...")
    print("Press Ctrl+C to stop.")
    try:
        test_model(env, predictor)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()