import tkinter as tk
import math

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

    def add_initial_nodes(self):
        # Left platform node
        x_left = 150  # Right edge of the left platform
        y_left = 525  # Vertically centered on the platform
        self.create_node_at_position(x_left, y_left, initial=True)

        # Right platform node
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
        line_id = self.canvas.create_line(x1, y1, x2, y2, fill="black", width=2)

        # Calculate the weight of the link
        length = math.hypot(x2 - x1, y2 - y1)
        link_weight = length * self.link_density

        # Store the link with weight
        self.links.append((node1_id, node2_id, line_id, link_weight))

        # Update connections
        self.nodes[node1_id]['connections'].append(node2_id)
        self.nodes[node2_id]['connections'].append(node1_id)

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
            for connected_node_id in node['connections']:
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

if __name__ == "__main__":
    root = tk.Tk()
    app = BridgeBuilder(root)
    root.mainloop()

    # After closing the GUI, calculate and print node loads
    node_loads = app.calculate_node_loads()
    for node_id, load in node_loads.items():
        print(f"Node {node_id}: Total Load = {load:.2f} grams")
