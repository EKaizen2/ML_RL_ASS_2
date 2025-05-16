from FourRooms import FourRooms
import numpy as np
import argparse

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.99, epsilon=0.2):
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
    parser = argparse.ArgumentParser(description='Run Scenario 3 with optional stochastic action space')
    parser.add_argument('-stochastic', action='store_true', 
                      help='Enable stochastic action space (20% random action probability)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create FourRooms Object with 'rgb' scenario and stochastic flag
    fourRoomsObj = FourRooms('rgb', stochastic=args.stochastic)
    
    # Initialize agent with modified parameters
    # If stochastic, increase epsilon for more exploration
    initial_epsilon = 0.3 if args.stochastic else 0.2
    agent = QLearningAgent(learning_rate=0.2, discount_factor=0.99, epsilon=initial_epsilon)
    
    # Training parameters
    # If stochastic, increase episodes and reduce epsilon decay rate
    num_episodes = 6000 if args.stochastic else 5000
    max_steps = 1000
    
    # Track best episode performance
    best_episode_steps = float('inf')
    best_episode_index = -1
    consecutive_successes = 0
    required_consecutive_successes = 3
    
    # Expected package collection order
    expected_order = [1, 2, 3]  # RED=1, GREEN=2, BLUE=3
    
    print(f"Starting training with {'stochastic' if args.stochastic else 'deterministic'} action space")
    
    for episode in range(num_episodes):
        # Start new epoch
        fourRoomsObj.newEpoch()
        current_pos = fourRoomsObj.getPosition()
        total_reward = 0
        steps = 0
        packages_collected = []  # Track order of collection
        last_package_step = 0
        
        while steps < max_steps:
            # Get action from agent
            # Include packages collected in state to help agent learn order
            state = (current_pos, len(packages_collected))  # Simplified state representation
            action = agent.get_action(state)
            
            # Take action
            gridType, new_pos, packages_remaining, is_terminal = fourRoomsObj.takeAction(action)
            
            # Calculate reward
            reward = -0.1  # Smaller step penalty
            
            # Check if a package was collected
            if gridType in [1, 2, 3]:  # Package found
                next_expected = expected_order[len(packages_collected)]
                if gridType == next_expected:
                    # Correct package order - exponential reward increase
                    reward = 50 * (2 ** len(packages_collected))  # 50, 100, 200 for each package
                    packages_collected.append(gridType)
                    last_package_step = steps
                else:
                    # Wrong package order
                    reward = -100  # Reduced penalty to encourage exploration
                    is_terminal = True
            
            # Add distance-based reward component
            if len(packages_collected) < 3 and steps - last_package_step > 50:
                reward -= 0.5  # Increased wandering penalty
            
            # Update Q-value with new state
            next_state = (new_pos, len(packages_collected))
            agent.update_q_value(state, action, reward, next_state)
            
            total_reward += reward
            current_pos = new_pos
            steps += 1
            
            # Check if all packages collected in correct order
            if len(packages_collected) == 3:  # All packages collected
                reward += 500  # Large bonus for completing the task
                if steps < best_episode_steps:
                    best_episode_steps = steps
                    best_episode_index = episode
                break
            
            if is_terminal:
                break
        
        # Update consecutive successes
        if len(packages_collected) == 3:  # Successfully collected all packages in order
            consecutive_successes += 1
            if consecutive_successes > 5:  # Reduce epsilon more quickly after consistent success
                # Slower epsilon decay for stochastic environment
                decay_rate = 0.98 if args.stochastic else 0.95
                agent.epsilon = max(0.05 if args.stochastic else 0.01, agent.epsilon * decay_rate)
        else:
            consecutive_successes = 0
            # Higher exploration ceiling for stochastic environment
            max_epsilon = 0.4 if args.stochastic else 0.3
            agent.epsilon = min(max_epsilon, agent.epsilon * 1.01)
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.1f}, "
                  f"Packages Collected: {len(packages_collected)}, Steps: {steps}")
            print(f"Best Path: Episode {best_episode_index}, Steps: {best_episode_steps}")
            print(f"Epsilon: {agent.epsilon:.3f}, Consecutive Successes: {consecutive_successes}")
    
    # Show the best path
    print(f"\nFinal Results:")
    print(f"Best path found in episode {best_episode_index}")
    print(f"Number of steps in best path: {best_episode_steps}")
    print(f"Environment type: {'Stochastic' if args.stochastic else 'Deterministic'}")
    fourRoomsObj.showPath(best_episode_index)

if __name__ == "__main__":
    main() 