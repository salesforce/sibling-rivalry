# Keeping Your Distance: Solving Sparse Reward TasksUsing Self-Balancing Shaped Rewards
Authors: Alexander Trott, Stephan Zheng, Caiming Xiong, and Richard Socher

Link to paper: https://arxiv.org/abs/1911.01417

This code provides an implementation of Sibling Rivalry and can be used to run the experiments presented in the paper.
Experiments are run using PyTorch (1.3.0) and make reference to OpenAI Gym.
In order to perform AntMaze experiments, you will need to have Mujoco installed (with a valid license).

## Running experiments
To run an experiment, use the following command:
```
python main.py --config-path <PATH.json> --log-dir <LOG DIRECTORY PATH> --dur <NUM EPOCHS TO TRAIN> --N <NUM WORKERS>
```
The type of experiment and algorithm are controlled by the settings provided in the config file (json).
Several example configuration files are provided in `sibling-rivalry/examples/`.
A directory corresponding to the name of the config file will be created inside of the log directory (second argument), and results will be stored there.

To train an agent on the 2D Point Maze using Sibling Rivalry, run:
```
python main.py --config-path examples/pointmaze/pointmaze-ppo-sr.json --log-dir <LOG DIRECTORY PATH> --dur 50 --N 20
```
Successful episodes should begin to occur around the 10th-15th training epoch.

The core implementation of Sibling Rivalry is found in `base/learners/distance.py`.
