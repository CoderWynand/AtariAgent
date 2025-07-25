# **Atari Agent ReadMe**



### **Introduction**

This project serves as an introduction to neural networks and machine learning, specifically reinforcement learning.



The aim of this project is to train an agent following the DQN algorithm to play Space Invaders, and then allow the user to play the agent in this environment. There is an existing model (that has been trained for approximately 1 500 000 steps) for users to test, or they can train it further. Additionally, they can choose to create a new agent to train from scratch.



### **Program overview**

##### Training the Agent

I make use of the Atari Learning Environment to train the agent to play Atari games. Stable Baselines3 provides the DQN implementation and training utilities (logging, monitoring, saving checkpoints).



If a new agent is being created, the desired properties(learning rate, frame skipping etc. ) must be changed around line 35 in TrainTheAgent.py.



Time steps for training must be specified around line 49.



Checkpoints are enabled every 50 000 steps, and training logs are saved for TensorBoard visualisation.



##### Playing the Agent

Once the PlayTheAgent.py script is run, it plays Space Invaders in a loop until the game window is manually closed by the user. 



### **Running the Program**

For proper functioning, this project requires: 

* Python version 11
* ALE version 0.8.1.
* A virtual environment is recommended for dependancy isolation.



*To activate the virtual environment (on Windows) run*: agent-environment\\Scripts\\activate



*Optionally, train the agent*: python TrainTheAgent.py



*Play using a saved agent*: python PlayTheAgent.py



Ensure that all dependencies from requirements.txt are installed in the virtual environment.



### **Known Issues/Future Improvements**



1. **Single agent availability:** Currently, the program only allows for one agent to be saved at a time, making it impossible to have different agents that were trained under different conditions at once.
2. **Lack of a GUI:** There is currently not a GUI implemented for this project, making it less user friendly.
3. **Simplicity of DQN algorithm:** The DQN algorithm was chosen for its simplicity and ease of use. Future improvements could look at implementing the Proximal Policy Optimisation algorithm, for faster learning speed.
