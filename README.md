# Chip-Placement-DDPG

This project implements a Deep Deterministic Policy Gradient (DDPG) agent to solve a chip placement problem. The environment simulates a grid where rectangular modules (components) are placed. The goal is to optimize the placement of these modules on the grid with minimal overlap and distance between them.

## Table of Contents
- [Introduction](#introduction)
- [Environment](#environment)
- [DDPG Agent](#ddpg-agent)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Visualization](#visualization)
- [Files](#files)

## Introduction
The chip placement problem involves placing a series of rectangular modules on a grid, with the objective of minimizing overlaps and the total distance between the centers of placed modules. The problem is solved using a reinforcement learning approach, specifically the DDPG algorithm.

## Environment
The environment simulates a grid of size 10000x10000. The environment supports four primary actions:

- **Insert:** Place a new module on the grid.
- **Delete:** Remove an existing module from the grid.
- **Swap:** Swap the positions of two placed modules.
- **Rotate:** Rotate a placed module by 90 degrees.

### Observations
The observation space consists of:

- The coordinates of the corners of the last placed module.
- The number of modules currently placed on the grid.

### Rewards
The reward is computed based on:

- **Total Distance:** Negative of the total distance between the centers of all placed modules.
- **Penalties:** Severe penalties are applied for invalid actions such as out-of-bounds placement, overlap, or unsuccessful module manipulation.

## DDPG Agent
The DDPG agent is responsible for learning the optimal placement strategy using a continuous action space. It consists of:

- **Actor Network:** Decides the next action to take (e.g., insert, delete, swap, rotate) and the associated parameters (e.g., coordinates for placement).
- **Critic Network:** Evaluates the action taken by the actor by computing a Q-value, guiding the actor during training.

The agent uses experience replay and a target network to stabilize training.

## Requirements
To run this project, you need Python 3.x and the following libraries:
- `numpy`
- `tensorflow`
- `matplotlib`

Install the dependencies using the provided `requirements.txt` file by running:
```bash
pip install -r requirements.txt
```

## Installation
1. Clone or download this repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies as described in the section.

## Usage
To run the environment and train the DDPG agent:

1. Make sure the component dimensions file (`Inputs.txt`) is correctly formatted and placed in the project directory.
2. Run the script:

    ```sh
    python chip_placement_ddpg.py
    ```

## Configuration
You can customize the following parameters directly in the script:

- `actor_lr`: Learning rate for the actor network.
- `critic_lr`: Learning rate for the critic network.
- `tau`: Soft update rate for the target networks.
- `gamma`: Discount factor for future rewards.
- `buffer_size`: Maximum size of the experience replay buffer.
- `batch_size`: Number of samples per training batch.

## Visualization
The environment supports visualization of the grid and placed modules. After training or during evaluation, you can plot the grid with the placed modules:

```python
env.plot_components(episode_num=1)
```

This will show a visual representation of the chip placement on the grid, with each module colored differently.

## Files
1. chip_placement_ddpg.py: Main script implementing the DDPG agent and chip placement environment.
2. requirements.txt: Lists the dependencies required to run the project.
3. Inputs.txt: Contains module dimensions for the chip placement environment (originally hp.scale.r). The script uses only the DIMENSIONS lines (format: DIMENSIONS x1 y1 x2 y2 x3 y3 x4 y4;), ignoring other sections like IOLIST and NETWORK.
