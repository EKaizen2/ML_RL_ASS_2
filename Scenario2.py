from FourRooms import FourRooms
import numpy as np

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
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

def main():
    # Create FourRooms Object with 'multi' scenario
    fourRoomsObj = FourRooms('multi')
    
    # Initialize agent
    agent = QLearningAgent()
    
    # Training parameters
    num_episodes = 1500  # Slightly increased episodes
    max_steps = 1000
    
    # Track best episode performance
    best_episode_packages = 0
    best_episode_steps = float('inf')
    best_episode_index = -1
    consecutive_successes = 0
    required_consecutive_successes = 5  # Number of consecutive successful episodes needed
    
    for episode in range(num_episodes):
        # Start new epoch
        fourRoomsObj.newEpoch()
        current_pos = fourRoomsObj.getPosition()
        total_reward = 0
        steps = 0
        packages_collected = 0
        last_package_step = 0  # Track when the last package was collected
        
        while steps < max_steps:
            # Get action from agent
            action = agent.get_action(current_pos)
            
            # Take action
            gridType, new_pos, packages_remaining, is_terminal = fourRoomsObj.takeAction(action)
            
            # Calculate reward
            reward = -1  # Base penalty for each step
            
            # Check if a package was collected
            if gridType in [1, 2, 3]:  # RED, GREEN, or BLUE package
                reward = 100 + (50 * packages_collected)  # Increasing reward for each subsequent package
                packages_collected += 1
                last_package_step = steps
            elif steps - last_package_step > 100:  # If no package collected in last 100 steps
                reward -= 1  # Additional penalty for wandering too long
            
            # Update Q-value
            agent.update_q_value(current_pos, action, reward, new_pos)
            
            total_reward += reward
            current_pos = new_pos
            steps += 1
            
            # Check if all packages are collected
            if packages_remaining == 0:
                # Update best path if this is the most efficient successful collection
                if packages_collected == 4 and steps < best_episode_steps:
                    best_episode_steps = steps
                    best_episode_index = episode
                    best_episode_packages = packages_collected
                break
            
            if is_terminal:
                break
        
        # Update consecutive successes
        if packages_collected == 4:  # All packages collected
            consecutive_successes += 1
        else:
            consecutive_successes = 0
        
        # Adjust epsilon based on performance
        if consecutive_successes >= required_consecutive_successes:
            agent.epsilon = max(0.01, agent.epsilon * 0.99)  # Reduce exploration
        else:
            agent.epsilon = min(0.1, agent.epsilon * 1.01)   # Increase exploration
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, "
                  f"Packages Collected: {packages_collected}, Steps: {steps}")
            print(f"Best Path: Episode {best_episode_index}, Steps: {best_episode_steps}")
            print(f"Consecutive Successful Episodes: {consecutive_successes}")
    
    # Show the best path instead of the last path
    print(f"\nFinal Results:")
    print(f"Best path found in episode {best_episode_index}")
    print(f"Number of steps in best path: {best_episode_steps}")
    fourRoomsObj.showPath(best_episode_index)

if __name__ == "__main__":
    main()