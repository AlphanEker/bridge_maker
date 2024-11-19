import tkinter as tk
import math
import numpy as np
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import random

try:
    import networkx as nx
except ImportError:
    print("NetworkX library is required for sturdiness calculation. Install it using 'pip install networkx'.")

MAX_NODES = 15

class BridgeBuilder:
    def __init__(self, root):
        self.root = root
        self.root.title("Bridge Builder")
        self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
        self.canvas.pack()

        # Platforms
        self.canvas.create_rectangle(50, 500, 150, 550, fill="gray")  # Left platform
        self.canvas.create_rectangle(650, 500, 750, 550, fill="gray")  # Right platform

        self.nodes = {}  # node_id: {'id': canvas_id, 'x': x, 'y': y, 'weight': weight, 'connections': []}
        self.links = []  # (node1_id, node2_id, line_id, weight)
        self.node_id_counter = 0
        self.selected_node = None

        # Define densities (weights per pixel length)
        self.node_weight = 1.0  # Weight per node (can be adjusted)
        self.link_density = 0.1  # grams per pixel

        # Add initial nodes at the edges of the platforms
        self.add_initial_nodes()

        # Bind events
        self.canvas.bind("<Button-1>", self.on_left_click)

        # Add "Start Optimization" button
        self.optimize_button = tk.Button(self.root, text="Start Optimization", command=self.start_optimization)
        self.optimize_button.pack(side=tk.BOTTOM)

        # Add "Run Evaluation" and "Stop Evaluation" buttons
        self.evaluate_button = tk.Button(self.root, text="Run Evaluation", command=self.start_evaluation)
        self.evaluate_button.pack(side=tk.BOTTOM)

        self.stop_button = tk.Button(self.root, text="Stop Evaluation", command=self.stop_evaluation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.BOTTOM)

        # Evaluation control flag
        self.evaluating = False

        # RL Agent placeholder
        self.rl_agent = None

    def start_evaluation(self):
        if self.rl_agent is None:
            print("Model is not trained yet. Please train the model before evaluation.")
            return

        # Disable other buttons to prevent conflicts
        self.optimize_button.config(state=tk.DISABLED)
        self.evaluate_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Set the evaluating flag
        self.evaluating = True

        # Start evaluation in a separate thread
        threading.Thread(target=self.run_evaluation).start()

    def stop_evaluation(self):
        self.evaluating = False
        self.stop_button.config(state=tk.DISABLED)
        self.evaluate_button.config(state=tk.NORMAL)
        self.optimize_button.config(state=tk.NORMAL)

    def run_evaluation(self):
        # Run the evaluation using the RL agent
        self.rl_agent.evaluate()
        # After evaluation, reset buttons
        self.stop_button.config(state=tk.DISABLED)
        self.evaluate_button.config(state=tk.NORMAL)
        self.optimize_button.config(state=tk.NORMAL)

    def add_initial_nodes(self):
        # Left platform node (ID 0)
        x_left = 150  # Right edge of the left platform
        y_left = 525  # Vertically centered on the platform
        self.create_node_at_position(x_left, y_left, initial=True)

        # Right platform node (ID 1)
        x_right = 650  # Left edge of the right platform
        y_right = 525
        self.create_node_at_position(x_right, y_right, initial=True)

    def create_node_at_position(self, x, y, initial=False):
        node_radius = 10
        canvas_id = self.canvas.create_oval(
            x - node_radius, y - node_radius, x + node_radius, y + node_radius,
            fill="red" if initial else "blue"
        )
        self.canvas.tag_bind(canvas_id, "<Button-1>", self.on_node_click)
        self.nodes[self.node_id_counter] = {
            'id': canvas_id,
            'x': x,
            'y': y,
            'weight': self.node_weight,
            'connections': []
        }
        self.node_id_counter += 1

    def on_left_click(self, event):
        # Check if clicked on empty space (not on a node)
        if self.canvas.find_withtag("current") == ():
            self.create_node(event)

    def create_node(self, event):
        x, y = event.x, event.y
        self.create_node_at_position(x, y)

    def on_node_click(self, event):
        # Get the node id from the canvas item clicked
        canvas_id = self.canvas.find_withtag("current")[0]
        node_id = self.get_node_id_by_canvas_id(canvas_id)
        if node_id is not None:
            if self.selected_node is None:
                # Select the node
                self.select_node(node_id)
            else:
                if node_id != self.selected_node:
                    # Connect the selected node with this node
                    if not self.is_connected(self.selected_node, node_id):
                        self.connect_nodes(self.selected_node, node_id)
                # Deselect the node
                self.deselect_node()

    def select_node(self, node_id):
        self.selected_node = node_id
        # Highlight the selected node
        self.canvas.itemconfig(self.nodes[node_id]['id'], outline="green", width=3)

    def deselect_node(self):
        # Remove highlight
        self.canvas.itemconfig(self.nodes[self.selected_node]['id'], outline="black", width=1)
        self.selected_node = None

    def connect_nodes(self, node1_id, node2_id):
        x1, y1 = self.nodes[node1_id]['x'], self.nodes[node1_id]['y']
        x2, y2 = self.nodes[node2_id]['x'], self.nodes[node2_id]['y']

        # Check if the line already exists
        if self.is_connected(node1_id, node2_id):
            return False

        line_id = self.canvas.create_line(x1, y1, x2, y2, fill="black", width=2)

        # Calculate the weight of the link
        length = math.hypot(x2 - x1, y2 - y1)
        link_weight = length * self.link_density

        # Store the link with weight
        self.links.append((node1_id, node2_id, line_id, link_weight))

        # Update connections
        self.nodes[node1_id]['connections'].append(node2_id)
        self.nodes[node2_id]['connections'].append(node1_id)
        return True

    def remove_link(self, node1_id, node2_id):
        if self.is_connected(node1_id, node2_id):
            # Find the link
            link_to_remove = None
            for link in self.links:
                if (link[0] == node1_id and link[1] == node2_id) or (link[0] == node2_id and link[1] == node1_id):
                    link_to_remove = link
                    break
            if link_to_remove:
                self.links.remove(link_to_remove)
                self.nodes[node1_id]['connections'].remove(node2_id)
                self.nodes[node2_id]['connections'].remove(node1_id)
                self.canvas.delete(link_to_remove[2])  # Remove the line from the canvas
                return True
        return False

    def get_node_id_by_canvas_id(self, canvas_id):
        for node_id, node in self.nodes.items():
            if node['id'] == canvas_id:
                return node_id
        return None

    def is_connected(self, node1_id, node2_id):
        return node2_id in self.nodes[node1_id]['connections']

    def get_environment(self):
        # Return nodes and links data
        nodes_data = {nid: {'x': n['x'], 'y': n['y']} for nid, n in self.nodes.items()}
        links_data = [(l[0], l[1]) for l in self.links]
        return nodes_data, links_data

    def calculate_node_loads(self):
        # Initialize total load on each node with its own weight + half the weight of connected links
        node_loads = {}
        for node_id, node in self.nodes.items():
            load = self.node_weight
            # Add half the weight of each connected link
            for link in self.links:
                if node_id == link[0] or node_id == link[1]:
                    load += link[3] / 2.0  # Half of the link's weight
            node_loads[node_id] = load

        # Build supporting nodes dictionary
        supporting_nodes = {node_id: [] for node_id in self.nodes.keys()}

        # For each node, find connected nodes below it (supporting nodes)
        for node_id, node in self.nodes.items():
            node_y = node['y']
            for connected_node_id in self.nodes[node_id]['connections']:
                connected_node_y = self.nodes[connected_node_id]['y']
                if connected_node_y > node_y:
                    # The connected node is below; it's a supporting node
                    supporting_nodes[node_id].append(connected_node_id)

        # Process nodes from top to bottom
        sorted_nodes = sorted(self.nodes.items(), key=lambda item: item[1]['y'])

        for node_id, node in sorted_nodes:
            total_load = node_loads[node_id]
            total_supports = len(supporting_nodes[node_id])
            if total_supports > 0:
                # Distribute the node's load equally among supporting nodes
                load_per_support = total_load / total_supports
                for support_node_id in supporting_nodes[node_id]:
                    node_loads[support_node_id] += load_per_support

        return node_loads

    # Additional methods for RL interaction

    def get_state(self):
        num_nodes = len(self.nodes)
        adjacency_matrix = np.zeros((MAX_NODES, MAX_NODES))
        positions = np.zeros(MAX_NODES * 2)

        # Fill in the adjacency matrix
        for link in self.links:
            node1_id, node2_id = link[0], link[1]
            adjacency_matrix[node1_id][node2_id] = 1
            adjacency_matrix[node2_id][node1_id] = 1

        # Fill in the positions
        for idx, node_id in enumerate(sorted(self.nodes.keys())):
            x = self.nodes[node_id]['x'] / self.canvas.winfo_width()
            y = self.nodes[node_id]['y'] / self.canvas.winfo_height()
            positions[idx * 2] = x
            positions[idx * 2 + 1] = y

        # Flatten adjacency matrix and concatenate positions
        state = np.concatenate([adjacency_matrix.flatten(), positions])

        return state

    def calculate_reward(self):
        node_loads = self.calculate_node_loads()
        load_values = list(node_loads.values())
        variance = np.var(load_values)
        total_weight = sum(load_values)
        sturdiness = self.calculate_sturdiness()

        # Use the continuous connectivity metric
        connectivity_reward = self.calculate_connectivity_metric()

        # Define weights for the reward components
        alpha = -1.0  # Adjusted variance penalty weight
        beta = -0.5  # Total weight penalty
        gamma = 3.0  # Sturdiness reward weight
        theta = 4.0 # Connectivity
        # Total reward
        reward = alpha * variance + beta * total_weight + gamma * sturdiness + theta * connectivity_reward

        return reward

    def is_bridge_connected(self):
        # Check if there's a path from start to end node
        start_node = 0  # Left platform node ID
        end_node = 1    # Right platform node ID

        visited = set()
        self.dfs(start_node, visited)
        return end_node in visited

    def calculate_connectivity_metric(self):
        try:
            G = nx.Graph()
            G.add_nodes_from(self.nodes.keys())
            G.add_edges_from([(link[0], link[1]) for link in self.links])
            if nx.has_path(G, 0, 1):
                path_length = nx.shortest_path_length(G, 0, 1)
                if path_length == 0:
                    connectivity_reward = 10  # Assign maximum reward when path_length is zero
                else:
                    connectivity_reward = 10 / path_length
            else:
                connectivity_reward = -100  # Penalty if no path exists
        except Exception as e:
            print(f"Exception in calculate_connectivity_metric: {e}")
            connectivity_reward = -100
        return connectivity_reward

    def dfs(self, node_id, visited):
        visited.add(node_id)
        for neighbor in self.nodes[node_id]['connections']:
            if neighbor not in visited:
                self.dfs(neighbor, visited)

    def calculate_sturdiness(self):
        try:
            G = nx.Graph()
            G.add_nodes_from(self.nodes.keys())
            G.add_edges_from([(link[0], link[1]) for link in self.links])
            sturdiness = nx.edge_connectivity(G, 0, 1)
        except Exception as e:
            print(f"Exception in calculate_sturdiness: {e}")
            sturdiness = 0
        return sturdiness

    def start_optimization(self):
        # Disable the optimize button to prevent multiple clicks
        self.optimize_button.config(state=tk.DISABLED)

        # Start the RL agent in a separate thread to prevent GUI freezing
        threading.Thread(target=self.run_rl_agent).start()

    def run_rl_agent(self):
        # Instantiate the RL agent
        self.rl_agent = RLAgent(self)

        # Run the optimization
        self.rl_agent.train(episodes=50)

        # Re-enable the optimize button
        self.optimize_button.config(state=tk.NORMAL)


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.output(x)
        return x


class RLAgent:
    def __init__(self, bridge_builder):
        self.bridge_builder = bridge_builder
        self.training = False  # Flag to indicate training is in progress

        num_nodes = len(self.bridge_builder.nodes)
        self.num_nodes = num_nodes

        self.state_size = MAX_NODES * MAX_NODES + MAX_NODES * 2
        self.action_size = MAX_NODES * MAX_NODES * 2  # For add and remove actions

        # Initialize neural network and optimizer
        self.model = DQN(self.state_size, self.action_size)
        self.target_model = DQN(self.state_size, self.action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # Hyperparameters
        self.gamma = 0.993  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.996

        # Experience replay memory
        self.memory = []
        self.batch_size = 64
        self.memory_capacity = 10000

        # Steps before updating the target network
        self.update_target_steps = 1000
        self.steps_done = 0

    def get_valid_actions(self):
        valid_actions = []
        num_nodes = self.num_nodes
        total_actions_per_type = num_nodes * num_nodes

        for action_type_index in range(2):  # 0: 'add', 1: 'remove'
            for node1_id in range(num_nodes):
                for node2_id in range(num_nodes):
                    if node1_id == node2_id:
                        continue  # Skip invalid node pairs

                    if action_type_index == 0:  # 'add' action
                        if not self.bridge_builder.is_connected(node1_id, node2_id):
                            action_index = self.encode_action(action_type_index, node1_id, node2_id)
                            valid_actions.append(action_index)
                    else:  # 'remove' action
                        if self.bridge_builder.is_connected(node1_id, node2_id):
                            action_index = self.encode_action(action_type_index, node1_id, node2_id)
                            valid_actions.append(action_index)
        return valid_actions

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # Store experiences in memory
        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        valid_actions = self.get_valid_actions()
        if not valid_actions:
            # No valid actions available
            return None

        if np.random.rand() <= self.epsilon:
            # Randomly select a valid action
            action = random.choice(valid_actions)
        else:
            # Predict action values
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.model(state_tensor).squeeze()

            # Mask invalid actions
            mask = torch.full((self.action_size,), float('-inf'))
            mask[valid_actions] = 0  # Set valid actions to 0 to keep their original values
            masked_action_values = action_values + mask

            # Select the action with the highest masked Q-value
            action = torch.argmax(masked_action_values).item()
        return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return  # Not enough samples to train

        minibatch = random.sample(self.memory, self.batch_size)

        state_batch = torch.FloatTensor([data[0] for data in minibatch])
        action_batch = torch.LongTensor([data[1] for data in minibatch]).unsqueeze(1)
        reward_batch = torch.FloatTensor([data[2] for data in minibatch])
        next_state_batch = torch.FloatTensor([data[3] for data in minibatch])
        done_batch = torch.FloatTensor([data[4] for data in minibatch])

        # Compute Q(s_t, a)
        q_values = self.model(state_batch).gather(1, action_batch).squeeze()

        # Compute Q(s_{t+1}, a)
        with torch.no_grad():
            next_q_values = self.target_model(next_state_batch).max(1)[0]

        # Compute target values
        target_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        # Compute loss
        loss = self.criterion(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decrease epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes):
        max_steps = 50  # Define maximum steps per episode
        for episode in range(episodes):
            state = self.bridge_builder.get_state().flatten()
            total_reward = 0

            for step in range(max_steps):
                action = self.choose_action(state)
                if action is None:
                    # No valid actions available; end the episode
                    print("No valid actions available. Ending episode early.")
                    break

                action_type, node1_id, node2_id = self.decode_action(action)
                if action_type is None:
                    # Invalid action decoding; skip this action
                    continue

                # Perform action
                success = False
                if action_type == 'add':
                    success = self.bridge_builder.connect_nodes(node1_id, node2_id)
                else:
                    success = self.bridge_builder.remove_link(node1_id, node2_id)

                # Ensure the action has been performed
                if not success:
                    reward = -10  # Penalty for invalid action
                    next_state = state
                else:
                    # Update GUI (if necessary)
                    self.bridge_builder.root.update_idletasks()
                    self.bridge_builder.root.update()

                    # Get reward and next state
                    reward = self.bridge_builder.calculate_reward()
                    next_state = self.bridge_builder.get_state().flatten()

                total_reward += reward

                # Store experience
                self.remember(state, action, reward, next_state, False)

                # Train the model
                self.replay()

                # Update state
                state = next_state

                self.steps_done += 1
                if self.steps_done % self.update_target_steps == 0:
                    self.update_target_model()

            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.2f}")

    def evaluate(self):
        self.model.eval()  # Set the model to evaluation mode
        self.epsilon = 0  # No exploration during evaluation
        max_steps = 100  # Define maximum steps for evaluation

        state = self.bridge_builder.get_state()
        total_reward = 0

        for step in range(max_steps):
            if not self.bridge_builder.evaluating:
                print("Evaluation stopped by user.")
                break

            action = self.choose_action(state)
            if action is None:
                print("No valid actions available. Ending evaluation.")
                break

            action_type, node1_id, node2_id = self.decode_action(action)
            if action_type is None:
                continue

            # Perform action
            if action_type == 'add':
                success = self.bridge_builder.connect_nodes(node1_id, node2_id)
            else:
                success = self.bridge_builder.remove_link(node1_id, node2_id)

            # Update GUI
            self.bridge_builder.root.update_idletasks()
            self.bridge_builder.root.update()

            # Get reward and next state
            reward = self.bridge_builder.calculate_reward()
            next_state = self.bridge_builder.get_state()

            total_reward += reward
            state = next_state

            # Sleep briefly to visualize changes
            self.bridge_builder.root.after(100)

        print(f"Evaluation completed. Total Reward: {total_reward}")
        self.model.train()  # Set the model back to training mode if needed

    def encode_action(self, action_type_index, node1_id, node2_id):
        total_actions_per_type = MAX_NODES * MAX_NODES
        node_pair_index = node1_id * MAX_NODES + node2_id
        action_index = action_type_index * total_actions_per_type + node_pair_index
        return action_index

    def decode_action(self, action_index):
        total_actions_per_type = MAX_NODES * MAX_NODES
        action_type_index = action_index // total_actions_per_type
        node_pair_index = action_index % total_actions_per_type

        node1_id = node_pair_index // MAX_NODES
        node2_id = node_pair_index % MAX_NODES

        if node1_id == node2_id or node1_id >= self.num_nodes or node2_id >= self.num_nodes:
            # Invalid action
            return None, None, None

        action_type = 'add' if action_type_index == 0 else 'remove'
        return action_type, node1_id, node2_id


if __name__ == "__main__":
    root = tk.Tk()

    # Initialize BridgeBuilder with random nodes
    app = BridgeBuilder(root)

    # Start the main GUI loop
    root.mainloop()
