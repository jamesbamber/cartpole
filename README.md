# Cartpole

A simple cart pole balancing physics simulation that supports reinforcement learning agents written in python.

## Instructions

### Running the simluation

To run the simluation, simply execute `run.py`.
Most of the simluation paramaters are configurable in `constants.py` (note that changing them might break the pre-trained reinforcement learning agents).

### Control types

The control type is selectable in `handle_action.py`, by uncommenting the desired control type in the `init` function. (Note that user_input is compatible with a reinforcement learning agent, simply uncomment both init lines).


### Game modes

There are currently two available modes, selectable in the `constants.py` file:
- `GAME_MODE` = 0 is a plain cartpole simulation.
- `GAME_MODE` = 1 is a multi-episode game, where the goal is to balance the pole for as long as possible.

