# Four Rooms Reinforcement Learning Assignment

Github Repository: https://github.com/EKaizen2/ML_RL_ASS_2.git 

## Program Structure

Makefile:  Includes commands for running all scenarios

### Core Files
- `FourRooms.py`: The main environment file that implements the Four Rooms grid world. It handles:
  - Grid world creation and management
  - Agent movement and package collection
  - State tracking and terminal conditions
  - Path visualization
  - Stochastic action space (20% random action probability)

### Scenario Files
1. `Scenario1.py`:
   - Implements Q-learning for a single package collection task
   - Uses 'simple' scenario from FourRooms
   - Command line option: `-stochastic` for random action probability

2. `Scenario2.py`:
   - Implements Q-learning for multiple package collection
   - Uses 'multi' scenario from FourRooms
   - Command line option: `-stochastic` for random action probability

3. `Scenario3.py`:
   - Implements Q-learning for ordered package collection (R->G->B)
   - Uses 'rgb' scenario from FourRooms
   - Command line option: `-stochastic` for random action probability

## Folders Experimentation in Scenario 1
- epsilon_paths - contains saved the paths of the epsilon greedy learning process 
- softmax_paths - contains saved the paths of the softmax learning process
- learnning - contains saved learning curves of both the epsilon greedy and softmax learning process, 
  with different learning rates and discount factors

## Report Scenario 1
This contains a 1 page analysis on the epsilon and softmax learning process show how different 
learning rates and discount factors affects the egents learning process - including visualization of learning curves.

## Running the Program

Each scenario can be run independently:

```bash
# Run Scenario 1 - To enable Stochastic process, pass a "-stochastic" as argument
python Scenario1.py [-stochastic] 

# Run Scenario 2
python Scenario2.py [-stochastic]

# Run Scenario 3
python Scenario3.py [-stochastic]
```

## Output
- Each scenario will display:
  - Training progress every 100 episodes
  - Best path found during training
  - Final results including number of steps
  - Visual representation of the best path

## Dependencies
- Python 3.x
- NumPy
- Matplotlib
