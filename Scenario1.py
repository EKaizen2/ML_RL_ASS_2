from FourRooms import FourRooms
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1, use_softmax=False):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.temperature = 1.0  # For softmax
        self.use_softmax = use_softmax
        self.q_table = {}
        
    def get_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0, 0]
            
        if self.use_softmax:
            # Softmax action selection
            q_values = np.array(self.q_table[state])
            exp_q = np.exp(q_values / self.temperature)
            probs = exp_q / np.sum(exp_q)
            return np.random.choice(4, p=probs)
        else:
            # Epsilon-greedy action selection
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
    parser = argparse.ArgumentParser(description='Run Scenario 1 with optional stochastic action space')
    parser.add_argument('-stochastic', action='store_true', 
                      help='Enable stochastic action space (20% random action probability)')
    return parser.parse_args()

def train_agent(fourRoomsObj, agent, num_episodes, max_steps, is_softmax=False):
    best_episode_steps = float('inf')
    best_episode_index = -1
    
    # Lists to track learning progress
    episode_rewards = []
    
    for episode in range(num_episodes):
        fourRoomsObj.newEpoch()
        current_pos = fourRoomsObj.getPosition()
        total_reward = 0
        steps = 0
        
        while steps < max_steps:
            state = current_pos
            action = agent.get_action(state)
            
            gridType, new_pos, packages_remaining, is_terminal = fourRoomsObj.takeAction(action)
            
            # Basic reward structure
            reward = -0.1  # Small penalty for each step
            if gridType > 0:  # Package found
                reward = 10.0
            
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
        
        # Update exploration parameters
        if is_softmax:
            agent.temperature = max(0.1, agent.temperature * 0.995)  # Decay temperature
        else:
            agent.epsilon = max(0.01, agent.epsilon * 0.995)  # Decay epsilon
        
        # Track learning progress
        episode_rewards.append(total_reward)
        
        if episode % 100 == 0:
            print(f"{'Softmax' if is_softmax else 'Epsilon-greedy'} - Episode {episode}, "
                  f"Total Reward: {total_reward:.1f}, Steps: {steps}")
            print(f"Best Path: Episode {best_episode_index}, Steps: {best_episode_steps}")
            print(f"{'Temperature' if is_softmax else 'Epsilon'}: "
                  f"{agent.temperature if is_softmax else agent.epsilon:.3f}")
    
    return best_episode_index, best_episode_steps, episode_rewards

def plot_total_reward_curve(epsilon_rewards, softmax_rewards, lr=0.2, gamma=0.95):
    plt.figure(figsize=(12, 6))
    plt.plot(epsilon_rewards, label='ε-greedy')
    plt.plot(softmax_rewards, label='Softmax')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Learning Curves (lr={lr}, γ={gamma})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    args = parse_arguments()
    
    # Create directories for saving paths if they don't exist
    os.makedirs('epsilon_paths', exist_ok=True)
    os.makedirs('softmax_paths', exist_ok=True)
    
    # Train epsilon-greedy agent
    print("\nTraining Epsilon-greedy Agent...")
    fourRoomsObj_epsilon = FourRooms('simple', stochastic=args.stochastic)
    epsilon_agent = QLearningAgent(learning_rate=0.1, discount_factor=0.95, epsilon=0.2, use_softmax=False)
    best_episode_epsilon, best_steps_epsilon, epsilon_rewards = train_agent(
        fourRoomsObj_epsilon, epsilon_agent, 1000, 1000, False)
    
    # Train softmax agent
    print("\nTraining Softmax Agent...")
    fourRoomsObj_softmax = FourRooms('simple', stochastic=args.stochastic)
    softmax_agent = QLearningAgent(learning_rate=0.1, discount_factor=0.95, epsilon=0.2, use_softmax=True)
    best_episode_softmax, best_steps_softmax, softmax_rewards = train_agent(
        fourRoomsObj_softmax, softmax_agent, 1000, 1000, True)
    
    # Show and save results
    print("\nFinal Results:")
    print("\nEpsilon-greedy Agent:")
    print(f"Best path found in episode {best_episode_epsilon}")
    print(f"Number of steps in best path: {best_steps_epsilon}")
    fourRoomsObj_epsilon.showPath(best_episode_epsilon)
    # fourRoomsObj_epsilon.showPath(best_episode_epsilon, savefig=f'epsilon_paths/best_path.png')
    
    print("\nSoftmax Agent:")
    print(f"Best path found in episode {best_episode_softmax}")
    print(f"Number of steps in best path: {best_steps_softmax}")
    fourRoomsObj_softmax.showPath(best_episode_softmax)
    # fourRoomsObj_softmax.showPath(best_episode_softmax, savefig=f'softmax_paths/best_path.png')
    
    # Plot only the total reward learning curve
    plot_total_reward_curve(epsilon_rewards, softmax_rewards, lr=0.2, gamma=0.95)
    
    print(f"\nEnvironment type: {'Stochastic' if args.stochastic else 'Deterministic'}")

if __name__ == "__main__":
    main()
