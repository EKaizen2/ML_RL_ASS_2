import numpy as np
from FourRooms import FourRooms
import matplotlib.pyplot as plt
import random
import os
import argparse

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        
    def get_action(self, state):
        # Initialize state if not in Q-table
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0, 0]
            
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)  # Random action
        return np.argmax(self.q_table[state])  # Best action
    
    def update_q_value(self, state, action, reward, next_state):
        # Initialize states if not in Q-table
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0, 0, 0]
            
        # Q-learning update
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + \
                    self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state][action] = new_value

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Scenario 1 with optional stochastic action space')
    parser.add_argument('-stochastic', action='store_true', 
                      help='Enable stochastic action space (20% random action probability)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create FourRooms Object with 'simple' scenario and stochastic flag
    fourRoomsObj = FourRooms('simple', stochastic=args.stochastic)
    
    # Initialize agent with modified parameters for stochastic environment if needed
    initial_epsilon = 0.2 if args.stochastic else 0.1
    agent = QLearningAgent(learning_rate=0.1, discount_factor=0.95, epsilon=initial_epsilon)
    
    # Training parameters
    num_episodes = 1000
    max_steps = 1000
    
    # Track best episode performance
    best_episode_steps = float('inf')
    best_episode_index = -1
    
    print(f"Starting training with {'stochastic' if args.stochastic else 'deterministic'} action space")
    
    for episode in range(num_episodes):
        # Start new epoch
        fourRoomsObj.newEpoch()
        current_pos = fourRoomsObj.getPosition()
        total_reward = 0
        steps = 0
        package_collected = False
        last_package_step = 0
        
        while steps < max_steps:
            # Get action from agent
            action = agent.get_action(current_pos)
            
            # Take action
            gridType, new_pos, packages_remaining, is_terminal = fourRoomsObj.takeAction(action)
            
            # Calculate reward
            reward = -0.1  # Small penalty for each step
            
            # Check if package was collected
            if gridType == 1:  # Package found
                reward = 100
                package_collected = True
            elif steps - last_package_step > 50:  # Wandering penalty
                reward -= 0.5
            
            # Update Q-value
            agent.update_q_value(current_pos, action, reward, new_pos)
            
            total_reward += reward
            current_pos = new_pos
            steps += 1
            
            # Check if package is collected
            if package_collected:
                if steps < best_episode_steps:
                    best_episode_steps = steps
                    best_episode_index = episode
                break
            
            if is_terminal:
                break
        
        # Adjust epsilon based on performance
        if package_collected:
            # Slower epsilon decay for stochastic environment
            decay_rate = 0.995 if args.stochastic else 0.99
            agent.epsilon = max(0.05 if args.stochastic else 0.01, agent.epsilon * decay_rate)
        else:
            # Higher exploration ceiling for stochastic environment
            max_epsilon = 0.3 if args.stochastic else 0.2
            agent.epsilon = min(max_epsilon, agent.epsilon * 1.01)
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.1f}, "
                  f"Package Collected: {package_collected}, Steps: {steps}")
            print(f"Best Path: Episode {best_episode_index}, Steps: {best_episode_steps}")
            print(f"Epsilon: {agent.epsilon:.3f}")
    
    # Show the best path
    print(f"\nFinal Results:")
    print(f"Best path found in episode {best_episode_index}")
    print(f"Number of steps in best path: {best_episode_steps}")
    print(f"Environment type: {'Stochastic' if args.stochastic else 'Deterministic'}")
    fourRoomsObj.showPath(best_episode_index)

if __name__ == "__main__":
    main()
