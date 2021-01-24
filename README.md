# **MEI-56307-2020-2021-1 Robotics Project Work: Reinforcement Learning In OpenAI Gym (Group_16: RL_G16)**

## Project Information

### Coaches
- Prof. Reza Ghabcheloo
- Nataliya Strokina

### Group Members:
- **Tarek Ibrahim (H293006):** programming (C/C++, Python, Matlab, Java), AI/ML, RL, Control, ROS, mechanics
- **Federico Dalla Rizza (H293016):** programming (.NET, Java, Python, R, Matlab), AI/ML, RL, ROS

## Setup & Run Steps:

### Option 1: As a Docker Image (Recommended):
TODO

### Option 2: Manual Installation From Source:
Prequisites: Python version 3.8.3. Also, optionally Anaconda for virtual environments and package management.

1. Clone this repository locally.
2. In a terminal, navigate to the Project directory.
3. Run `pip install -r requirements.txt`
4. If you have GPU and wish to use it with this program, you have to install the appropriate PyTorch GPU version yourself from: https://pytorch.org/get-started/locally/
5. [Install ffmpeg on windows](https://www.wikihow.com/Install-FFmpeg-on-Windows) or Run: `sudo apt-get install xvfb` on Linux.
5. Clone the repository "[gym-custom]()" locally .
6. In a terminal, navigate to the directory which contains the repo.
7. Run `pip install -e gym-custom`
8. Check and edit program inputs (if needed) in main.py
9. Run main.py (from terminal by `python main.py [-<short_flag>=<value>] [--long_flag=<value>]`). For help with the flags/arguments: Run `python main.py -h|--help`

## Directories:

1. **Models:** Saved Models (after training & validation)
2. **Output:** Program outputs (plots, video demos, etc)
3. **Materials:** Other supporting material (gifs, pseudocodes, etc)

## Files:

1. **main.py**: A high-level abstraction layer for running train.py over a sweep of the hyperparameter space while extracting the best model based on validation results, plotting the results of the hyperparameter space exploration and running test.py on the best models
2. **train.py**: Initialization, Training and Validation of Own & Stable-baselines3 AC models on training environments. Own algorithm is implemented here.
3. **test.py**: Testing of Own & Stable-baselines3 AC models, a PID model, and random model on the testing environments 
4. **models.py**: Includes own \[NN\] models, inncluding network model of own AC implementation
5. **utils.py**: Auxiliary functions for validation, testing, hyperparameter optimization, plotting, etc.

## Description:

(Final project documentation will be later detailed in a report)
Up-to-date Project Report Documentation: https://www.overleaf.com/read/grjvqfbhmbkz

![AC Controller Implementation on CartPole demo](Project/materials/demo.gif)

#### Project Aim: 

Goal is to study RL through playing around with OpenAI gym environments as a part of MEI-56307-2020-2021-1 Robotics Project Work course in Tampere University.

#### Environments:

The chosen OpenAI Gym environment is "[CartPole-v1](https://gym.openai.com/envs/CartPole-v1/)" which represents an instance of the problem of balancing and inverted pendulum through control of the base (cart) movement atop which the pendulum/pole is mounted. The force supplied to the cartpole dynamics has been modified from discrete to continous (Box type), such that it's value is in the bounds [-10,10]. To ensure a better continuity, the force is adjusted by a smoothing factor proportional to the difference between the force supplied at the previous timestep and the one calculated for the current timestep.

1. **Trainig environments:** Further adjusted such that env0 corresponds to default environment provided by OpenAI gym, env1 corresponds to a case where rewards are linearly proportional to the angle deviation from 0 error, and env2 corresponds to the case where rewards are exponentially proportional.
2. **Testing environments**: Have the same reward definition as the default case provided by OpenAI gym, but the kinematic parameters of the cartpole is arbitrarily modified such that env0 corresponds to the default environment, evn1 corresponds to a case where parameters are larger, and env2 to a case where parameters are smaller.

##### Method:

- In design of the program, considerations for conducting fair expirements to get a true unbiased insight on the true performance of own implementation are given a high priority.
- In choosing the algorithm to solve the problem, considerations included state space type (continous vs discrete), action space type, time horizon (episodic vs continous), presence of a knwon environment model (here the model is assumed unkown to justify use of RL algorithms).
- In choosing the network architecture, several other design options have been tried.
- Discrete control signals are replaced by continous ones and smoothing is attempted as that could be more realistically implemented in real life
- Own implementation is compared to the stable-baselines3 implementation (acts as an upper bound on performance) and a random agent (acts as a lower bound).
- Implemented algorithms:
- [X] Actor-Critic
- [X] Model-based Controller (e.g. PID)

##### Implementation:

- Program inputs are supplied in/to main.py, then a sweep of the hyperparameter space takes place, where the models (own and SB) are trained, validated and saved (in models dir.) via train.py with each combination of the hyperparameters and training environments. Training results and plots are saved to output/resutls_and_plots
- The best models (own and SB) are determined based on highest average reward according to validation
- Hyperparameter exploration results are plotted
- The best models are used for testing in test.py on the testing environments, where overall results are also extracted and a random agent performance is added to the comparison. Here the generalization capabilities of the model are tested.

### Extending the Project

Plans for future work is listed in the TODOs of main.py. The project currently is not optimized for extensions through external collaboration.

### Contacts:

For suggestions, questions, requests or bug reporting please contact:
Tarek Ibrahim (tarek.ibrahim@tuni.fi)

### Source & Helpful Links:
- https://gym.openai.com/
- https://stable-baselines.readthedocs.io/en/master/guide/rl.html (Not compatible with Tensorflow 2.0 or above)
- For stable baselines with Pytorch use stable-baselines3: https://stable-baselines3.readthedocs.io/en/master/guide/rl.html 
- A Specialization (multiple related courses) on RL: https://www.coursera.org/specializations/reinforcement-learning#courses
- Implementation of CartPole Dynamics: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
- Classic Control Projects in OpenAI Gym (5 projects, 3 of which are discussed in the RL specialization, and one of which is the CartPole): https://github.com/openai/gym/tree/master/gym/envs/classic_control
- Similar problem to CartPole (Inverted Pendulum) discussed in an RL course of the specialization: https://www.coursera.org/learn/prediction-control-function-approximation/home/week/4