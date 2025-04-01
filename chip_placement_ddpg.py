
#Chip Placement Environment with DDPG Agent

#This module defines a custom chip placement environment using reinforcement learning,
#implemented using the Deep Deterministic Policy Gradient (DDPG) algorithm. The environment 
#manages the placement of rectangular modules on a 10,000 x 10,000 grid, allowing for 
#insertion, deletion, swapping, and rotation actions. The DDPG agent learns to optimize 
#module placement by minimizing distances between placed modules while avoiding overlaps 
#and out-of-bounds errors.

#Key Classes:
#- OUNoise: Implements the Ornstein-Uhlenbeck process for generating noise for action exploration.
#- ChipPlacementEnv: Custom environment for placing chip components on a grid.
#- DDPGAgent: Deep Deterministic Policy Gradient agent that interacts with the ChipPlacementEnv.


import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.patches as patches
import matplotlib.colors as mcolors

# Define constants
GRID_SIZE = 10000  # Define the size of the grid (10000x10000)
ACTION_SPACE = 4  # Define the number of actions: inserting, deleting, swapping, and rotating modules
OBSERVATION_SPACE = 9  # Define the size of the observation space (x1, y1, x2, y2, x3, y3, x4, y4 for each module + count of placed modules)

# Ornstein-Uhlenbeck process for generating noise
class OUNoise:

    
    #Implements the Ornstein-Uhlenbeck process for generating temporally correlated noise.
    #Used to add exploration noise to the actions of the DDPG agent.
    
   # Attributes:
   #action_dim (int): Dimension of the action space.
   #mu (float): Mean of the noise.
   #theta (float): Speed of mean reversion.
   #sigma (float): Scale of the noise.


    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        # Initialize the noise parameters
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Resets the internal state of the noise to the mean value."""
        # Reset the internal state to the mean value
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        """Updates the internal state with noise and returns the new state."""
        # Update the internal state with noise
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

# Define file path for component data
INPUT_FILE = "C:/Users/Shreya/PycharmProjects/Inputs.txt"  

# Function to load components from a file
def load_components(file_path):

    
    # Loads component data from the specified file.
    # Args:
    # file_path (str): Path to the file containing component dimensions.
    # Returns:
    # list: A list of tuples where each tuple contains a sequential ID and coordinates of a module.
    

    components = []
    try:
        with open(file_path, "r") as f:
            print("File opened successfully.")
            lines = f.readlines()
            sequential_id = 1  # Start sequential ID from 1
            for idx, line in enumerate(lines):
                line = line.strip()
                if line.startswith("DIMENSIONS"):
                    # Extract coordinates from the line
                    parts = line[len("DIMENSIONS"):].strip().strip(";").split()
                    try:
                        coords = list(map(int, parts))
                        if len(coords) == 8:
                            component = (sequential_id, coords)  # Use sequential_id
                            components.append(component)
                            print(f"Loaded DIMENSIONS: {component}")
                            sequential_id += 1  # Increment sequential ID
                        else:
                            print(f"Error: Invalid number of coordinates in line '{line}'")
                    except ValueError as e:
                        print(f"Error parsing line '{line}': {e}")
    except FileNotFoundError:
        print("Error: File not found.")
    return components

# Class to represent the chip placement environment
class ChipPlacementEnv:

    

# Custom environment for placing chip components on a grid.

# This environment allows for the insertion, deletion, swapping, and rotation of rectangular modules
#on a 10,000 x 10,000 grid. The goal is to place modules in a way that minimizes the distance between
#them while avoiding overlaps and out-of-bounds errors.

#Attributes:
#    grid (np.ndarray): A 2D numpy array representing the grid where modules are placed.
#    placed_modules (list): A list of tuples representing the placed modules and their coordinates.
#    components (list): A list of tuples representing the available modules for placement.
#    action_space (int): The number of available actions.
#    observation_space (int): The size of the observation space.


    def __init__(self, components):
        # Initialize the grid and other environment parameters
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)  # Initialize the grid with zeros
        self.placed_modules = []  # List to store placed modules
        self.components = components[:]  # Copy the initial components
        self.current_step = 0
        self.max_steps = 100  # Maximum steps per episode
        self.max_modules = len(components)  # Maximum number of modules to place (based on the number of components)
        self.action_space = ACTION_SPACE
        self.observation_space = OBSERVATION_SPACE
        self.initial_components = components[:]  # Store initial components for reset
        self.component_colors = {}  # Dictionary to store component colors
        self._initialize_colors()  # Initialize component colors
        self.deletion_count = 0  # Track number of deletions per episode

        self.last_failed_action = None  # Track the last failed action

    def reset(self):
        # Reset the environment to its initial state
        self.grid.fill(0)  # Reset the grid
        self.placed_modules = []  # Clear placed modules
        self.current_step = 0
        self.components = self.initial_components[:]  # Reset components to initial state
        self.deletion_count = 0  # Reset deletion count
        print("Environment reset.")
        return self._get_observation()

    def _get_observation(self):
        # Get the current observation of the environment
        if not self.placed_modules:
            return np.zeros(OBSERVATION_SPACE)
        module = self.placed_modules[-1][1][:OBSERVATION_SPACE - 1]  # Get coordinates of the last placed module
        return np.array(module + [len(self.placed_modules)]).flatten()

    def step(self, action):
        # Take a step in the environment based on the given action
        self.current_step += 1
        action_choice = np.argmax(action[:4])  # Determine the action choice (insert, delete, swap, rotate)
        print(f"Step {self.current_step}, action {action_choice}")

        prev_state = self._get_observation().copy()
        reward = -self._calculate_total_distance()  # Reward based on the total distance between modules

        success = False
        penalty = 0

        # Perform the chosen action
        if action_choice == 0:
            success, penalty = self.insert_module(action[4:])
        elif action_choice == 1 and len(self.placed_modules) > 0 and self.deletion_count < 3:
            success = self.delete_module()
        elif action_choice == 2:
            success = self.swap_modules()
        elif action_choice == 3:
            success = self.rotate_module(action[4:])

        if not success:
            reward -= 10
            self.last_failed_action = action_choice  # Update last failed action
            print(f"Action {action_choice} failed, penalizing.")
        else:
            reward += 20
            self.last_failed_action = None  # Reset last failed action if the action succeeds
            print(f"Action {action_choice} succeeded, rewarding.")

        reward -= penalty  # Apply penalty if there was any overlap or out-of-bounds attempt

        new_state = self._get_observation()
        done = self.current_step >= self.max_steps

        return new_state, reward, done, {}

    # Insert module based on action parameters
    def insert_module(self, action_params):
        if not self.components:
            print("No more components to place.")
            return False, 0

        module_id, module = random.choice(self.components)  # Choose a random module to place
        x1, y1, x2, y2, x3, y3, x4, y4 = module

        # Calculate the width and height of the module
        width = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
        height = max(y1, y2, y3, y4) - min(y1, y2, y3, y4)

        print(f"Attempting to insert module {module_id} with width: {width}, height: {height}")

        # Determine the offset for placement
        x_offset = int((action_params[0] + 1) / 2 * (GRID_SIZE - width))
        y_offset = int((action_params[1] + 1) / 2 * (GRID_SIZE - height))

        # Calculate the new coordinates after applying the offset
        new_coords = [(x1 + x_offset, y1 + y_offset), (x2 + x_offset, y2 + y_offset),
                      (x3 + x_offset, y3 + y_offset), (x4 + x_offset, y4 + y_offset)]

        # Check for out-of-bounds placement
        if any(x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE for x, y in new_coords):
            print(f"Failed to insert module {module_id}: Out of bounds.")
            return False, 1000  # Apply severe penalty for out-of-bounds placement

        # Check if all new coordinates are valid and unoccupied
        if all(self._can_place(x, y) for x, y in new_coords):
            # Place the module on the grid
            self.placed_modules.append((module_id, [x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset,
                                                    x3 + x_offset, y3 + y_offset, x4 + x_offset, y4 + y_offset]))
            for x, y in new_coords:
                self.grid[x, y] = 1
            self.components = [c for c in self.components if c[0] != module_id]  # Remove the placed module
            print(f"Inserted module {module_id} at position: {new_coords}")
            return True, 0

        print(f"Failed to insert module {module_id}: Overlap detected.")
        return False, 1000  # Apply severe penalty for overlap

    # Delete a randomly chosen module
    def delete_module(self):
        if len(self.placed_modules) == 0:
            return False, 100

        idx = random.choice(range(len(self.placed_modules)))
        module_id, module = self.placed_modules[idx]
        x1, y1, x2, y2, x3, y3, x4, y4 = module
        if all(self.grid[x, y] == 1 for x, y in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]):
            # Remove the module from the grid
            self.placed_modules.pop(idx)
            for x, y in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]:
                self.grid[x, y] = 0
            self.components.append((module_id, [x1, y1, x2, y2, x3, y3, x4, y4]))  # Re-add the deleted module
            self.deletion_count += 1  # Increment deletion count
            print(f"Deleted module {module_id} at position: {[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]}")
            return True
        print(f"Failed to delete module {module_id}: Module not found on grid.")
        return False, 100

    # Swap two randomly chosen modules
    def swap_modules(self):
        if len(self.placed_modules) < 2:
            print("Not enough modules to swap.")
            return False
        idx1, idx2 = random.sample(range(len(self.placed_modules)), 2)
        module1_id, module1 = self.placed_modules[idx1]
        module2_id, module2 = self.placed_modules[idx2]
        # Swap the modules
        self.placed_modules[idx1], self.placed_modules[idx2] = self.placed_modules[idx2], self.placed_modules[idx1]
        print(f"Swapped module {module1_id} with module {module2_id}")
        return True

    # Rotate a randomly chosen module
    def rotate_module(self, action_params):
        if not self.placed_modules:
            print("No modules to rotate.")
            return False

        module_id, module = random.choice(self.placed_modules)
        x1, y1, x2, y2, x3, y3, x4, y4 = module
        print(f"Attempting to rotate module {module_id}")

        # Rotation logic (90 degrees clockwise for simplicity)
        cx = (x1 + x2 + x3 + x4) / 4
        cy = (y1 + y2 + y3 + y4) / 4
        new_coords = [(int(cy - (y - cy)), int(cx + (x - cx))) for x, y in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]]

        # Check for out-of-bounds placement
        if any(x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE for x, y in new_coords):
            print(f"Failed to rotate module {module_id}: Out of bounds.")
            return False, 1000  # Apply severe penalty for out-of-bounds rotation

        # Check if all new coordinates are valid and unoccupied
        if all(self._can_place(x, y) for x, y in new_coords):
            # Apply the rotation
            rotated_module = [coord for point in new_coords for coord in point]
            self.placed_modules = [(id_, rotated_module if id_ == module_id else coords) for id_, coords in self.placed_modules]
            for (old_x, old_y), (new_x, new_y) in zip([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], new_coords):
                self.grid[old_x, old_y] = 0
                self.grid[new_x, new_y] = 1
            print(f"Rotated module {module_id} to new position: {new_coords}")
            return True

        print(f"Failed to rotate module {module_id}: Overlap detected.")
        return False, 1000  # Apply severe penalty for overlap

    # Calculate the total distance between the centers of all placed modules
    def _calculate_total_distance(self):
        if len(self.placed_modules) < 2:
            return 0
        total_distance = 0
        for i in range(len(self.placed_modules) - 1):
            for j in range(i + 1, len(self.placed_modules)):
                module1_id, module1 = self.placed_modules[i]
                module2_id, module2 = self.placed_modules[j]
                # Calculate the center of each module
                module1_center = np.mean(np.array(module1).reshape(4, 2), axis=0)
                module2_center = np.mean(np.array(module2).reshape(4, 2), axis=0)
                # Calculate the distance between the centers
                total_distance += np.linalg.norm(module1_center - module2_center)
        return total_distance

    # Check if a coordinate is within grid bounds and unoccupied
    def _can_place(self, x, y):
        if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
            print(f"Coordinate out of bounds: ({x}, {y})")
            return False
        if self.grid[x, y] != 0:
            print(f"Grid cell occupied: ({x}, {y})")
            return False
        return True

    # Render the grid (visualization)
    def render(self, mode='human'):
        plt.imshow(self.grid, cmap='gray')
        plt.show()

    # Initialize component colors for visualization
    def _initialize_colors(self):
        available_colors = list(mcolors.CSS4_COLORS.values())
        random.shuffle(available_colors)  # Shuffle to ensure variety
        color_index = 0

        for component_id, component in self.components:
            if color_index >= len(available_colors):
                color_index = 0  # Reset the index to cycle through colors
            color = available_colors[color_index]
            self.component_colors[component_id] = color
            color_index += 1

    # Plot the components on the grid
    def plot_components(self, episode_num=None):
        fig, ax = plt.subplots()
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.set_facecolor('white')  # Set background color to white

        for module_id, module in self.placed_modules:
            x1, y1, x2, y2, x3, y3, x4, y4 = module
            x_coords = [x1, x2, x3, x4, x1]
            y_coords = [y1, y2, y3, y4, y1]
            color = self.component_colors.get(module_id, 'black')
            ax.add_patch(patches.Polygon(xy=list(zip(x_coords, y_coords)), fill=True, edgecolor='black', facecolor=color, alpha=0.6))

        # Add legend with updated component numbering and place it outside the grid
        handles = [patches.Patch(color=color, label=f'Component {comp_id}') for comp_id, color in self.component_colors.items()]
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.title(f'Chip Components Placement (Episode {episode_num})' if episode_num is not None else 'Chip Components Placement')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)
        plt.show()

# DDPG Agent class
class DDPGAgent:
    # This block defines the DDPGAgent class, which implements the Deep Deterministic Policy Gradient (DDPG) algorithm.
    # The agent interacts with an environment, using an actor-critic architecture. The actor network selects actions 
    # based on the current state, while the critic network evaluates the action-value function. The agent also uses 
    # target networks for stable training and a replay buffer for experience replay, enhancing learning efficiency and stability.

    def __init__(self, env, actor_lr=0.001, critic_lr=0.002, tau=0.005, gamma=0.99, buffer_size=100000, batch_size=64):
        # Initialize the DDPG agent with the given parameters
        self.env = env
        self.state_dim = env.observation_space
        self.action_dim = env.action_space + 2  # 4 for action choice + 2 for x and y parameters
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.actor_model = self.create_actor_model()  # Create the actor model
        self.target_actor_model = self.create_actor_model()  # Create the target actor model
        self.critic_model = self.create_critic_model()  # Create the critic model
        self.target_critic_model = self.create_critic_model()  # Create the target critic model

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)  # Actor optimizer
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)  # Critic optimizer

        self.replay_buffer = deque(maxlen=self.buffer_size)  # Replay buffer for experience replay
        self.noise = OUNoise(self.action_dim)  # Ornstein-Uhlenbeck noise for action exploration

    # Create the actor model
    def create_actor_model(self):
        state_input = layers.Input(shape=(self.state_dim,))
        h1 = layers.Dense(400, activation='relu')(state_input)
        h2 = layers.Dense(300, activation='relu')(h1)
        output = layers.Dense(self.action_dim, activation='tanh')(h2)
        model = tf.keras.Model(inputs=state_input, outputs=output)
        return model

    # Create the critic model
    def create_critic_model(self):
        state_input = layers.Input(shape=(self.state_dim,))
        state_h1 = layers.Dense(16, activation='relu')(state_input)
        state_h2 = layers.Dense(32, activation='relu')(state_h1)

        action_input = layers.Input(shape=(self.action_dim,))
        action_h1 = layers.Dense(32, activation='relu')(action_input)

        concat = layers.Concatenate()([state_h2, action_h1])
        h1 = layers.Dense(400, activation='relu')(concat)
        h2 = layers.Dense(300, activation='relu')(h1)
        output = layers.Dense(1)(h2)

        model = tf.keras.Model(inputs=[state_input, action_input], outputs=output)
        return model

    # Update target model weights
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    # Policy function to select actions
    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor_model(state))  # Sample actions from the actor model
        noise = self.noise.evolve_state()  # Add noise for exploration
        sampled_actions = sampled_actions.numpy() + noise

        # Ensure actions are within the valid range
        legal_action = np.clip(sampled_actions, -1, 1)

        # Check if the last action failed and avoid it
        if self.env.last_failed_action is not None:
            # Create a mask to set the failed action probability to a very low value
            action_probs = np.ones(self.env.action_space)
            action_probs[self.env.last_failed_action] = 1e-6
            action_probs /= action_probs.sum()  # Normalize the probabilities

            # Select an action using the masked probabilities
            action_choice = np.random.choice(range(self.env.action_space), p=action_probs)
            legal_action[:4] = -1  # Initialize to invalid range
            legal_action[action_choice] = 1  # Set the chosen action to valid range

        return [np.squeeze(legal_action)]

    # Train the agent
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        minibatch = random.sample(self.replay_buffer, self.batch_size)

        state_batch = np.array([transition[0] for transition in minibatch])
        action_batch = np.array([transition[1] for transition in minibatch])
        reward_batch = np.array([transition[2] for transition in minibatch])
        next_state_batch = np.array([transition[3] for transition in minibatch])
        done_batch = np.array([transition[4] for transition in minibatch])

        next_actions = self.target_actor_model(next_state_batch)  # Compute target actions
        future_rewards = self.target_critic_model([next_state_batch, next_actions])  # Compute target Q-values
        updated_q_values = reward_batch + self.gamma * future_rewards * (1 - done_batch)  # Compute updated Q-values

        with tf.GradientTape() as tape:
            q_values = self.critic_model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(updated_q_values - q_values))  # Critic loss

        critic_grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)  # Compute gradients
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))  # Apply gradients

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch)
            critic_value = self.critic_model([state_batch, actions])
            actor_loss = -tf.math.reduce_mean(critic_value)  # Actor loss

        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)  # Compute gradients
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))  # Apply gradients

        # Update target networks
        self.update_target(self.target_actor_model.variables, self.actor_model.variables, self.tau)
        self.update_target(self.target_critic_model.variables, self.critic_model.variables, self.tau)

    # Add experience to the replay buffer
    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

# Training function
# This block defines the training function for a DDPG agent in a custom environment. 
# The agent is trained over a specified number of episodes. In each episode, the environment is reset, 
# and the agent interacts with it by selecting actions based on its policy, storing experiences in a replay buffer, 
# and updating its networks. The function also includes initial random actions and visualizes the placement of components 
# after each episode, providing feedback on the agent's performance.
def train_ddpg(agent, env, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()  # Reset the environment at the start of each episode
        episode_reward = 0

        # Insert all initial components before taking random actions
        for _ in range(12):  # Insert 12 initial components
            action = np.random.uniform(-1, 1, env.action_space + 2)  # Include space for action params
            env.insert_module(action[4:])  # Only pass the action parameters (x, y)

        while True:
            action = agent.policy(tf.convert_to_tensor([state], dtype=tf.float32))  # Generate an action using the policy
            next_state, reward, done, _ = env.step(action[0])  # Take a step in the environment
            agent.add_to_replay_buffer(state, action[0], reward, next_state, done)  # Store the experience
            agent.train()  # Train the agent
            state = next_state
            episode_reward += reward

            if done:  # Check if the episode is done
                break

        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")  # Print the reward for the episode
        env.plot_components(episode_num=episode + 1)  # Plot the components
        print(f"Number of components placed: {len(env.placed_modules)}")  # Print the number of components placed

if __name__ == "__main__":
    components = load_components(INPUT_FILE)  # Load components from the file
    env = ChipPlacementEnv(components)  # Create the environment
    agent = DDPGAgent(env)  # Create the agent

    num_episodes = 30  # Define the number of training episodes
    train_ddpg(agent, env, num_episodes)  # Train the agent
