import numpy as np
from FourRooms import FourRooms
import matplotlib.pyplot as plt
import random

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_strategy='epsilon_greedy'):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_strategy = exploration_strategy
        
        # Initialize Q-table
        # State space: (x, y, packages_remaining)
        # Action space: UP, DOWN, LEFT, RIGHT
        self.q_table = {}
        
        # Exploration parameters
        self.epsilon = 1.0  # For ε-greedy
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.temperature = 1.0  # For softmax
        self.temperature_decay = 0.995
        self.temperature_min = 0.1

    def get_state_key(self, position, packages_remaining):
        return (position[0], position[1], packages_remaining)

    def get_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in range(4)}

        if self.exploration_strategy == 'epsilon_greedy':
            if random.random() < self.epsilon:
                return random.choice(range(4))
            return max(self.q_table[state].items(), key=lambda x: x[1])[0]
        
        else:  # softmax
            q_values = [self.q_table[state][action] for action in range(4)]
            exp_q = np.exp(np.array(q_values) / self.temperature)
            probabilities = exp_q / np.sum(exp_q)
            return np.random.choice(range(4), p=probabilities)

    def update(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0 for action in range(4)}
        
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        
        # Q-learning update
        new_value = (1 - self.learning_rate) * old_value + \
                    self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state][action] = new_value

    def decay_exploration(self):
        if self.exploration_strategy == 'epsilon_greedy':
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
            self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)

def train_agent(exploration_strategy, num_episodes=1000):
    env = FourRooms('simple')
    agent = QLearningAgent(exploration_strategy=exploration_strategy)
    episode_rewards = []
    
    for episode in range(num_episodes):
        env.newEpoch()
        total_reward = 0
        steps = 0
        
        while not env.isTerminal():
            current_state = agent.get_state_key(env.getPosition(), env.getPackagesRemaining())
            action = agent.get_action(current_state)
            
            # Take action and get reward
            cell_type, new_pos, packages_remaining, is_terminal = env.takeAction(action)
            
            # Reward function
            reward = -0.1  # Small negative reward for each step
            if cell_type > 0:  # Found package
                reward = 10.0
            
            next_state = agent.get_state_key(new_pos, packages_remaining)
            agent.update(current_state, action, reward, next_state)
            
            total_reward += reward
            steps += 1
            
            if steps > 1000:  # Prevent infinite loops
                break
        
        agent.decay_exploration()
        episode_rewards.append(total_reward)
    
    return env, episode_rewards

def main():
    # Train with epsilon-greedy
    env_epsilon, rewards_epsilon = train_agent('epsilon_greedy')
    
    # Save epsilon-greedy path
    plt.figure()
    env_epsilon.showPath(-1)
    plt.savefig('.epsilon_paths/epsilon_greedy_path.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Train with softmax
    env_softmax, rewards_softmax = train_agent('softmax')
    
    # Save softmax path
    plt.figure()
    env_softmax.showPath(-1)
    plt.savefig('.softmax_paths/softmax_path.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot learning curves
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_epsilon, label='ε-greedy', alpha=0.7)
    plt.plot(rewards_softmax, label='Softmax', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curves for Different Exploration Strategies')
    plt.legend()
    plt.savefig('.learning_curves/learning_curves.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
