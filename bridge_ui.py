import tkinter as tk

class BridgeBuilder:
    def __init__(self, root):
        self.root = root
        self.root.title("Bridge Builder")
        self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
        self.canvas.pack()

        # Platforms
        self.canvas.create_rectangle(50, 500, 150, 550, fill="gray")  # Left platform
        self.canvas.create_rectangle(650, 500, 750, 550, fill="gray")  # Right platform

        self.nodes = {}  # Store node_id: {'id': canvas_id, 'x': x, 'y': y}
        self.links = []  # Store tuples of (node1_id, node2_id, canvas_line_id)
        self.node_id_counter = 0
        self.selected_node = None

        # Bind events
        self.canvas.bind("<Button-1>", self.create_node)
        self.canvas.bind("<Button-3>", self.delete_node)
        self.canvas.bind("<Button-2>", self.connect_nodes)

    def create_node(self, event):
        x, y = event.x, event.y
        node_radius = 10
        canvas_id = self.canvas.create_oval(
            x - node_radius, y - node_radius, x + node_radius, y + node_radius, fill="blue"
        )
        self.nodes[self.node_id_counter] = {'id': canvas_id, 'x': x, 'y': y}
        self.node_id_counter += 1

    def delete_node(self, event):
        node_id = self.find_node(event.x, event.y)
        if node_id is not None:
            # Delete the node
            self.canvas.delete(self.nodes[node_id]['id'])
            del self.nodes[node_id]
            # Delete associated links
            self.links = [
                link for link in self.links if node_id not in link[:2]
            ]

    def connect_nodes(self, event):
        node_id = self.find_node(event.x, event.y)
        if node_id is not None:
            if self.selected_node is None:
                self.selected_node = node_id
            else:
                # Create link if not already connected
                if not self.is_connected(self.selected_node, node_id):
                    x1, y1 = self.nodes[self.selected_node]['x'], self.nodes[self.selected_node]['y']
                    x2, y2 = self.nodes[node_id]['x'], self.nodes[node_id]['y']
                    line_id = self.canvas.create_line(x1, y1, x2, y2, fill="black")
                    self.links.append((self.selected_node, node_id, line_id))
                self.selected_node = None

    def find_node(self, x, y):
        for node_id, node in self.nodes.items():
            node_x, node_y = node['x'], node['y']
            if (node_x - x) ** 2 + (node_y - y) ** 2 <= 100:  # Radius squared
                return node_id
        return None

    def is_connected(self, node1_id, node2_id):
        for link in self.links:
            if (node1_id, node2_id) == link[:2] or (node2_id, node1_id) == link[:2]:
                return True
        return False

    def get_environment(self):
        # Return nodes and links data
        nodes_data = {nid: {'x': n['x'], 'y': n['y']} for nid, n in self.nodes.items()}
        links_data = [(l[0], l[1]) for l in self.links]
        return nodes_data, links_data

if __name__ == "__main__":
    root = tk.Tk()
    app = BridgeBuilder(root)
    root.mainloop()
