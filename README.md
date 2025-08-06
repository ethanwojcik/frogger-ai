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
