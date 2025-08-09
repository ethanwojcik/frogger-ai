# Frogger-AI

This is a fork of existing frogger project by [rhrlima](https://github.com/rhrlima) that you can check out [Here](https://github.com/rhrlima/frogger). My goal was to use the existing code and add logic for deep learning to allow an agent learn to avoid obstacles and score points. I have made simplifications to the state of the game, such as making it grid based to allow discrete positions, which significantly shrinks the range of possible inputs to be more understandable by a neural net. This makes it less "human-playable" but easier for a machine learning algorithim to pick up. I made each episode a "life" aka until a frog dies. I plan to add more features such as logs, statistic tracking, more stage customization and user control as well as model improvement methods such as fine tuning.

## Getting Started

Instructions needed to run the project.

### Prerequisites

This code requires Python3 and Pygame to run.

### Installing

* [Python3](https://www.python.org/download/releases/3.0/) - Python Website.

* [Pip](https://pip.pypa.io/en/stable/installing/) - Package Installer for Python.

* [Pygame](https://www.pygame.org/news) - The Pygame library website.

Install packages by running the following
```
sudo apt-get install python3-tk
pip install -r requirements.txt
```

### Running

To run the game, call:

```
python frogger.py
```

### Game Instructions

The game consists of 3 screens. The hyperparamater setup, the game board setup, and the game. Additionally a matplotlib window will open up to track rolling accuracy, and overall accuracy. Accuracy is measured by a "succesfull" iteration, which can be configured in hyperparamater setup. The default value is 0, representing an iteration that did not die immediately.


## Hyperparamater Setup
All values decreased with left arrow, increased with right arrow. Cycle through paramaters with up and down arrow keys.
 * gamma: Discount factor for future rewards. Higher gamma means agent values future rewards more, while lower focus on immediate rewards. (0 < gamma ≤ 1)
 * epsilon: Starting value for epsilon in epsilon-greedy exploration. For each iteration, a random # between 1 and 0 is generated. If it is less than epsilon, we choose exploration. Otherwise, we choose a best known action from memory. High epsilon indicates high chance of exploration.
 * epsilon_decay: Rate at which epsilon decreases after each episode.
 * epsilon_min: Minimum value epsilon will reach. At this point, the best choice should be picked frequently, with a small chance for exploration.
 * lr: Learning rate - controls how much neural net weights are updated during training.
 * batch_size: Number of experiences sampled from memory for each training step. Higher values require more memory and computation.
 * memory_size: Maximum number of experiences stored in the replay buffer. Higher value uses more memory.
 * inner_layer_size: Number of neurons in hidden layers of the Neural net. Larger size allows more complex pattern recognition, but increases computation and risk of overfitting.
 * episodes: Total number of training iterations to run
 * reload_freq: How often (in steps, not episodes) the target net is updated from the main network.
 * penalty_per_tick: The penatly subtracted from score each time step. Encourages speed.
 * use_prev_model: Boolean indicating whether a previous model will be used. Only works when model sizes are same, which depends on number of obstacles


## Game Board Setup Controls
Allows setup and saving of configs. A config consists of a board with obstacles and a finish line. However a config may also contain no obstacles and no finish line. The 0th config contains the default, and cannot be overwritten as it should be used to create new configs. Note that obstacles can be placed "outside" of the board game and still be valid. 
* Left Click: Place Obstacle
* L: Set obstacle as log
* C: Set obstacle as car
* F: Set finish line in selected row. Only 1 finish line may exist at a time
* Left Arrow Key: Decrease size of obstacle by 1
* Right Arrow Key: Increase size of obstacle by 1
* Up Arrow Key: Increase speed of obstacle by 1
* Down Arrow Key: Decrease speed of obstacle by 1
* N: Next config
* P: Previous config
* V: View currently selected config. Will change board to match selected config from json file
* A: Add config. Current board will be written to a new config file, but currently selected config will not change.
* S: Save / overwrite current board to currently selected config.
* Enter: Finish Setup

## Game Controls:
* Up Arrow Key: Increases ticks per second
* Down Arrow Key: Decreases ticks per second
* M: Sets ticks per second to max. Useful for maximum speed
* D: Sets ticks per second to 1. Useful when carefully analyzing behavior
* Spacebar: Pauses game, setting ticks per second to 0. Press again to resume game with ticks per second at 1.
* S: Saves agent model to file
* L: Loads agent model from file. Make sure previously saved model is of same size (ie has exact same # of obstacles)
