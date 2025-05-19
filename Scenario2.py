from FourRooms import FourRooms
import numpy as np
import argparse

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        
    def get_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0, 0]
            
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)
        return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0, 0, 0]
            
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + \
                    self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state][action] = new_value

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Scenario 2 with optional stochastic action space')
    parser.add_argument('-stochastic', action='store_true', 
                      help='Enable stochastic action space (20% random action probability)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    fourRoomsObj = FourRooms('multi', stochastic=args.stochastic)
    
    # Adjust learning parameters for stochastic environment
    if args.stochastic:
        agent = QLearningAgent(
            learning_rate=0.05,  # Slower learning rate for more stable learning
            discount_factor=0.99,
            epsilon=0.4  # Higher initial exploration
        )
        num_episodes = 3000  # More episodes for stochastic case
    else:
        agent = QLearningAgent(
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon=0.2
        )
        num_episodes = 2000
    
    max_steps = 1000
    best_episode_steps = float('inf')
    best_episode_index = -1
    consecutive_successes = 0
    
    print(f"Starting training with {'stochastic' if args.stochastic else 'deterministic'} action space")
    
    for episode in range(num_episodes):
        fourRoomsObj.newEpoch()
        current_pos = fourRoomsObj.getPosition()
        total_reward = 0
        steps = 0
        packages_collected = 0
        
        while steps < max_steps:
            state = current_pos
            action = agent.get_action(state)
            
            gridType, new_pos, packages_remaining, is_terminal = fourRoomsObj.takeAction(action)
            
            # Enhanced reward structure with stochastic consideration
            reward = -0.1  # Small penalty for each step
            
            if gridType > 0:  # Package found
                reward = 30.0  # Higher reward for finding a package in stochastic case
                packages_collected += 1
            
            # Additional reward for collecting all packages
            if packages_remaining == 0:
                reward = 100.0  # Higher completion reward
                consecutive_successes += 1
            else:
                consecutive_successes = 0
            
            # Update Q-value
            agent.update_q_value(state, action, reward, new_pos)
            
            total_reward += reward
            current_pos = new_pos
            steps += 1
            
            if is_terminal:
                if packages_remaining == 0 and steps < best_episode_steps:
                    best_episode_steps = steps
                    best_episode_index = episode
                break
        
        # Adaptive epsilon decay based on performance and environment type
        if args.stochastic:
            if consecutive_successes >= 3:  # After consistent success
                agent.epsilon = max(0.1, agent.epsilon * 0.995)  # Slower decay
            else:
                agent.epsilon = min(0.4, agent.epsilon * 1.005)  # Slower increase
        else:
            if packages_collected == 3:
                agent.epsilon = max(0.01, agent.epsilon * 0.995)
            else:
                agent.epsilon = min(0.3, agent.epsilon * 1.01)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.1f}, Steps: {steps}, Packages: {packages_collected}")
            print(f"Best Path: Episode {best_episode_index}, Steps: {best_episode_steps}")
            print(f"Epsilon: {agent.epsilon:.3f}, Consecutive Successes: {consecutive_successes}")
    
    print(f"\nFinal Results:")
    print(f"Best path found in episode {best_episode_index}")
    print(f"Number of steps in best path: {best_episode_steps}")
    print(f"Environment type: {'Stochastic' if args.stochastic else 'Deterministic'}")
    fourRoomsObj.showPath(best_episode_index)

if __name__ == "__main__":
    main()