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
    num_episodes = 1000
    max_steps = 1000
    
    for episode in range(num_episodes):
        # Start new epoch
        fourRoomsObj.newEpoch()
        current_pos = fourRoomsObj.getPosition()
        total_reward = 0
        steps = 0
        packages_collected = 0
        
        while steps < max_steps:
            # Get action from agent
            action = agent.get_action(current_pos)
            
            # Take action
            gridType, new_pos, packages_remaining, is_terminal = fourRoomsObj.takeAction(action)
            
            # Calculate reward
            reward = -1  # Small penalty for each step
            
            # Check if a package was collected
            if gridType in [1, 2, 3]:  # RED, GREEN, or BLUE package
                reward = 100
                packages_collected += 1
            
            # Update Q-value
            agent.update_q_value(current_pos, action, reward, new_pos)
            
            total_reward += reward
            current_pos = new_pos
            steps += 1
            
            # Check if all packages are collected
            if packages_remaining == 0:
                break
            
            if is_terminal:
                break
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, "
                  f"Packages Collected: {packages_collected}, Steps: {steps}")
    
    # Show final path
    fourRoomsObj.showPath(-1)

if __name__ == "__main__":
    main()