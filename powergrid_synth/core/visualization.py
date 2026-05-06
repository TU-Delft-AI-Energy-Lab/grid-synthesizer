"""
The visualization module contains various methods for plots. 
"""
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from matplotlib.widgets import Button
from typing import Dict, Any, Tuple, List, Optional

class MatplotlibDropdown:
    """
    A custom Dropdown widget for Matplotlib, created by combining Buttons.
    """
    def __init__(self, fig, rect, labels, active=0, on_select=None):
        self.fig = fig
        self.labels = labels
        self.active_idx = active
        self.on_select = on_select
        self.is_menu_open = False
        self.buttons = []
        
        # Dimensions
        left, bottom, width, height = rect
        self.rect = rect
        self.height = height
        
        # 1. Create the Main Button (Header)
        # Position: The top of the defined rect
        ax_main = plt.axes([left, bottom, width, height])
        self.main_btn = Button(ax_main, labels[active], color='#e0e0e0', hovercolor='#d0d0d0')
        self.main_btn.on_clicked(self.toggle_menu)
        
        # 2. Create Option Buttons (Hidden by default)
        # They will appear BELOW the main button
        self.option_axes = []
        self.option_btns = []
        
        for i, label in enumerate(labels):
            # Calculate position: stacked downwards
            # y = bottom - (i+1)*height
            y_pos = bottom - ((i + 1) * height)
            ax_opt = plt.axes([left, y_pos, width, height])
            ax_opt.set_visible(False) # Hide initially
            
            btn = Button(ax_opt, label, color='white', hovercolor='#f0f0f0')
            # We need to bind the specific index/label to the callback
            # Using a default argument in lambda captures the current value of i/label
            btn.on_clicked(lambda event, idx=i, lbl=label: self.select_option(idx, lbl))
            
            self.option_axes.append(ax_opt)
            self.option_btns.append(btn)

    def toggle_menu(self, event):
        self.is_menu_open = not self.is_menu_open
        for ax in self.option_axes:
            ax.set_visible(self.is_menu_open)
        self.fig.canvas.draw_idle()

    def select_option(self, idx, label):
        # Update state
        self.active_idx = idx
        self.main_btn.label.set_text(label)
        
        # Close menu
        self.is_menu_open = False
        for ax in self.option_axes:
            ax.set_visible(False)
            
        self.fig.canvas.draw_idle()
        
        # Trigger callback
        if self.on_select:
            self.on_select(label)

class GridVisualizer:
    """
    Visualization module for synthetic power grids.
    Allows plotting the grid with different layouts including Yifan Hu, Kamada-Kawai, and Voltage Layered.
    """

    def __init__(self):
        # Default colormap for voltage levels
        # Fix for Matplotlib 3.7+ deprecation of cm.get_cmap
        if hasattr(matplotlib, 'colormaps'):
            self.cmap = matplotlib.colormaps['tab10']
        else:
            self.cmap = cm.get_cmap('tab10')
            
        # Keep references to widgets to prevent garbage collection
        self._widgets = []

    def _get_node_colors(self, graph: nx.Graph) -> List[Any]:
        """Assigns colors to nodes based on their 'voltage_level' attribute."""
        colors = []
        for node in graph.nodes():
            level = graph.nodes[node].get('voltage_level', 0)
            colors.append(self.cmap(level))
        return colors

    def _get_layered_layout(self, graph: nx.Graph) -> Dict[int, np.ndarray]:
        """Custom layout: Places nodes in horizontal bands based on voltage level."""
        pos = {}
        levels = {}
        for node, data in graph.nodes(data=True):
            lvl = data.get('voltage_level', 0)
            if lvl not in levels:
                levels[lvl] = []
            levels[lvl].append(node)
        
        max_width = max([len(nodes) for nodes in levels.values()]) if levels else 1
        
        for lvl, nodes in levels.items():
            y = -lvl * 10 
            x_values = np.linspace(-max_width/2, max_width/2, len(nodes))
            for i, node in enumerate(nodes):
                pos[node] = np.array([x_values[i], y])
        return pos

    def _yifan_hu_layout(self, G: nx.Graph, iterations: int = 100, k: Optional[float] = None) -> Dict[int, np.ndarray]:
        """Implementation of the Yifan Hu force-directed layout algorithm."""
        nodes = list(G.nodes())
        n = len(nodes)
        if n == 0: return {}
        
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        adj_matrix = nx.to_scipy_sparse_array(G, nodelist=nodes, format='csr')
        pos = np.random.rand(n, 2) * 2 - 1
        
        if k is None:
            k = 1.0 / np.sqrt(n) if n > 0 else 1.0
        
        step = n / 10.0
        
        for it in range(iterations):
            delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist_sq = np.sum(delta**2, axis=2)
            dist = np.sqrt(dist_sq)
            dist[dist == 0] = 0.001
            
            fr = (k**2 / dist_sq[..., np.newaxis]) * delta
            np.fill_diagonal(fr[:, :, 0], 0)
            np.fill_diagonal(fr[:, :, 1], 0)
            
            displacement = np.sum(fr, axis=1)
            
            rows, cols = adj_matrix.nonzero()
            for u, v in zip(rows, cols):
                if u >= v: continue
                delta_uv = pos[v] - pos[u]
                dist_uv = np.linalg.norm(delta_uv)
                if dist_uv == 0: continue
                fa_mag = dist_uv / k
                fa_vec = fa_mag * (delta_uv / dist_uv)
                displacement[u] += fa_vec
                displacement[v] -= fa_vec

            length = np.linalg.norm(displacement, axis=1)
            length[length == 0] = 0.1
            scale = np.minimum(step, length) / length
            pos += displacement * scale[:, np.newaxis]
            step *= 0.95 

        return {nodes[i]: pos[i] for i in range(n)}

    def _find_tree_root(self, tree: nx.Graph) -> Any:
        """Find the best root for a tree: prefer lowest voltage_level, then highest degree."""
        best = None
        best_key = (float('inf'), -1)
        for n, d in tree.nodes(data=True):
            vl = d.get('voltage_level', float('inf'))
            deg = tree.degree(n)
            key = (vl, -deg)
            if key < best_key:
                best_key = key
                best = n
        return best

    def _hierarchical_tree_layout(self, graph: nx.Graph, layer_sep: float = 1.0,
                                   node_sep: float = 1.0) -> Dict[Any, np.ndarray]:
        """Hierarchical top-down layout suitable for tree/forest graphs.
        
        Root at the top, children below. Handles disconnected forests by
        placing each component side-by-side.
        """
        pos = {}
        x_offset = 0.0

        for component_nodes in nx.connected_components(graph):
            tree = graph.subgraph(component_nodes)
            if len(tree) == 0:
                continue

            root = self._find_tree_root(tree)
            # BFS to assign layers
            layers: Dict[int, List] = {}
            visited = {root}
            queue = [(root, 0)]
            parent_map = {}
            while queue:
                node, depth = queue.pop(0)
                layers.setdefault(depth, []).append(node)
                for nbr in tree.neighbors(node):
                    if nbr not in visited:
                        visited.add(nbr)
                        parent_map[nbr] = node
                        queue.append((nbr, depth + 1))

            # Assign x positions bottom-up (leaves first) for balanced spacing
            node_x: Dict[Any, float] = {}
            max_depth = max(layers.keys()) if layers else 0

            for d in range(max_depth, -1, -1):
                for node in layers[d]:
                    children = [nbr for nbr in tree.neighbors(node)
                                if parent_map.get(nbr) == node]
                    if not children:
                        # Leaf: assign next available x
                        node_x[node] = len(node_x) * node_sep
                    else:
                        # Parent: center above children
                        child_xs = [node_x[c] for c in children]
                        node_x[node] = (min(child_xs) + max(child_xs)) / 2.0

            # Shift so component starts at x_offset
            min_x = min(node_x.values()) if node_x else 0
            for node in node_x:
                node_x[node] -= min_x
                node_x[node] += x_offset

            for node in tree.nodes():
                depth = next(d for d, nodes in layers.items() if node in nodes)
                pos[node] = np.array([node_x[node], -depth * layer_sep])

            max_x = max(node_x.values()) if node_x else 0
            x_offset = max_x + 2.0 * node_sep  # gap between components

        return pos

    def _radial_tree_layout(self, graph: nx.Graph, radius_step: float = 1.0) -> Dict[Any, np.ndarray]:
        """Radial layout suitable for tree/forest graphs.
        
        Root at the center, each BFS layer on a concentric circle.
        Handles disconnected forests by rotating each component into its own sector.
        """
        components = list(nx.connected_components(graph))
        n_comp = len(components)
        if n_comp == 0:
            return {}

        # Each component gets an angular sector
        sector_size = 2 * np.pi / n_comp
        pos = {}

        for ci, component_nodes in enumerate(components):
            tree = graph.subgraph(component_nodes)
            if len(tree) == 0:
                continue

            root = self._find_tree_root(tree)
            # BFS layers
            layers: Dict[int, List] = {}
            visited = {root}
            queue = [(root, 0)]
            parent_map = {}
            while queue:
                node, depth = queue.pop(0)
                layers.setdefault(depth, []).append(node)
                for nbr in tree.neighbors(node):
                    if nbr not in visited:
                        visited.add(nbr)
                        parent_map[nbr] = node
                        queue.append((nbr, depth + 1))

            # Angular sector for this component
            angle_start = ci * sector_size
            angle_end = angle_start + sector_size

            # Assign angular positions bottom-up
            node_angle: Dict[Any, float] = {}
            max_depth = max(layers.keys()) if layers else 0
            # Count leaves to distribute them evenly in the sector
            leaves = [n for n in tree.nodes()
                      if all(parent_map.get(nbr) != n for nbr in tree.neighbors(n))
                      or tree.degree(n) == 1 and n != root]
            # Actually, simpler: leaves are nodes with no children in BFS tree
            leaf_nodes = []
            for n in tree.nodes():
                children = [nbr for nbr in tree.neighbors(n) if parent_map.get(nbr) == n]
                if not children and n != root:
                    leaf_nodes.append(n)
            if not leaf_nodes and len(tree) == 1:
                leaf_nodes = [root]

            n_leaves = max(len(leaf_nodes), 1)
            leaf_angles = np.linspace(angle_start, angle_end, n_leaves, endpoint=False)
            leaf_idx = 0

            for d in range(max_depth, -1, -1):
                for node in layers[d]:
                    children = [nbr for nbr in tree.neighbors(node)
                                if parent_map.get(nbr) == node]
                    if not children:
                        node_angle[node] = leaf_angles[leaf_idx % n_leaves]
                        leaf_idx += 1
                    else:
                        child_angles = [node_angle[c] for c in children]
                        node_angle[node] = np.mean(child_angles)

            for node in tree.nodes():
                depth = next(d for d, nodes in layers.items() if node in nodes)
                r = depth * radius_step
                theta = node_angle.get(node, angle_start)
                pos[node] = np.array([r * np.cos(theta), r * np.sin(theta)])

        return pos
    
    def _draw_edges_impedance(self, ax, grid, pos, alpha=0.8):
        """Helper to draw edges colored by impedance."""
        edges = []
        z_vals = []
        for u, v, d in grid.edges(data=True):
            edges.append((u, v))
            z_vals.append(d.get('z', 0.0))
        
        if edges:
            edge_collection = nx.draw_networkx_edges(
                grid, pos,
                edgelist=edges,
                edge_color=z_vals,
                edge_cmap=plt.cm.coolwarm,
                width=2.0,
                alpha=alpha,
                ax=ax
            )
            return edge_collection
        return None

    def plot_grid(self, graph: nx.Graph, layout: str = 'kamada_kawai', title: str = "Grid", show_labels: bool = False, 
                 show_bus_types: bool = False, show_impedance: bool = False, figsize: Tuple[int, int] = (12, 10)):
        """
        Static plot function for grid topology.
        Options allow overlaying bus types or impedance features.
        """
        plt.figure(figsize=figsize)
        
        if show_bus_types:
            # If bus types are requested, delegate to the more complex handler
            # Use 'best' location for adaptive placement
            self._draw_bus_types_on_ax(plt.gca(), graph, layout, title, legend_loc='best', legend_bbox=None, show_impedance=show_impedance)
        else:
            self._draw_graph_on_ax(plt.gca(), graph, layout, title, show_labels, legend_loc='best', show_impedance=show_impedance)
            
        plt.tight_layout()
        plt.show()

    def plot_subgraphs(self, grid: nx.Graph, layout: str = 'kamada_kawai', title: str = "Subgraphs by Voltage Level", 
                       show_impedance: bool = False, figsize: Tuple[int, int] = (15, 5)):
        """
        Plots subgraphs for each voltage level side-by-side (max 3 per row).
        
        Args:
            grid (nx.Graph): The main power grid graph.
            layout (str): Layout algorithm to use.
            title (str): Main title for the figure.
            show_impedance (bool): Whether to color edges by impedance.
            figsize (Tuple[int, int]): Base size for the figure (width, height for one row). 
                                       Height will scale with the number of rows.
        """
        # Identify levels
        levels = sorted(list(set(nx.get_node_attributes(grid, 'voltage_level').values())))
        n_plots = len(levels)
        
        if n_plots == 0:
            print("No voltage levels found in grid.")
            return

        # Calculate grid dimensions (max 3 cols)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
        
        # Adjust figsize based on rows
        base_w, base_h = figsize
        final_figsize = (base_w, base_h * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=final_figsize)
        
        # Standardize axes to a list/flat array
        if n_plots == 1:
            axes_flat = [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()
        else:
            axes_flat = axes.flatten()
            
        # Iterate and plot
        for i, level in enumerate(levels):
            ax = axes_flat[i]
            
            # Extract nodes for this level
            nodes = [n for n, d in grid.nodes(data=True) if d.get('voltage_level') == level]
            subgraph = grid.subgraph(nodes)
            
            sub_title = f"Level {level} ({len(nodes)} nodes)"
            
            # Use existing helper
            self._draw_graph_on_ax(ax, subgraph, layout, sub_title, show_labels=False, 
                                   legend_loc='best', show_impedance=show_impedance)
            
        # Hide empty subplots if any
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis('off')
            
        if title:
            # Adjust title position based on number of rows
            plt.suptitle(title, y=1.02 if n_rows > 1 else 1.05, fontsize=16)
            
        plt.tight_layout()
        plt.show()

    def plot_load_gen_bubbles(self, grid: nx.Graph, layout: str = 'kamada_kawai', title: str = "Generation vs Load", 
                            show_impedance: bool = False, figsize: Tuple[int, int] = (12, 10)):
        """
        Bubble plot showing generation and load magnitudes.
        Generators are blue squares, Loads are red circles.
        Size is proportional to capacity/load.
        Optionally plots impedance on edges.
        """
        plt.figure(figsize=figsize)
        ax = plt.gca()
        
        # 1. Layout
        if layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(grid)
        elif layout == 'yifan_hu':
            pos = self._yifan_hu_layout(grid)
        elif layout == 'spring':
             pos = nx.spring_layout(grid, seed=42)
        elif layout == 'voltage_layered':
            pos = self._get_layered_layout(grid)
        elif layout == 'hierarchical_tree':
            pos = self._hierarchical_tree_layout(grid)
        elif layout == 'radial_tree':
            pos = self._radial_tree_layout(grid)
        else:
             pos = nx.kamada_kawai_layout(grid)
             
        # 2. Draw Edges
        if show_impedance:
            edge_coll = self._draw_edges_impedance(ax, grid, pos)
            if edge_coll:
                # Horizontal Colorbar at Bottom
                plt.colorbar(edge_coll, ax=ax, label="Impedance (Z)", orientation='horizontal', fraction=0.046, pad=0.04)
        else:
            nx.draw_networkx_edges(grid, pos, alpha=0.2, ax=ax)
        
        # 3. Draw Loads (Red circles)
        load_nodes = [n for n, d in grid.nodes(data=True) if d.get('bus_type') == 'Load']
        if load_nodes:
            # Scale factor for visibility
            load_sizes = [grid.nodes[n].get('pl', 0) * 2 for n in load_nodes]
            nx.draw_networkx_nodes(grid, pos, nodelist=load_nodes, node_color='red', 
                                   node_size=load_sizes, alpha=0.6, label='Load', ax=ax)
        
        # 4. Draw Gens (Blue squares)
        gen_nodes = [n for n, d in grid.nodes(data=True) if d.get('bus_type') == 'Gen']
        if gen_nodes:
            # Scale factor for visibility
            gen_sizes = [grid.nodes[n].get('pg', 0) * 2 for n in gen_nodes]
            nx.draw_networkx_nodes(grid, pos, nodelist=gen_nodes, node_color='blue', 
                                   node_shape='s', node_size=gen_sizes, alpha=0.6, label='Gen (Dispatched)', ax=ax)
        
        # 5. Create Manual Legend
        # Using fixed size markers (markersize=8) instead of scaling with data
        legend_elements = [
            mlines.Line2D([], [], color='red', marker='o', linestyle='None', 
                          markersize=8, label='Load', alpha=0.6),
            mlines.Line2D([], [], color='blue', marker='s', linestyle='None', 
                          markersize=8, label='Gen (Dispatched)', alpha=0.6)
        ]
        
        # Adaptive Legend Placement
        ax.legend(handles=legend_elements, loc='best')
        ax.axis('off')
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        plt.show()

    def _draw_bus_types_on_ax(self, ax, graph: nx.Graph, layout_name: str, title: str, 
                              legend_loc='center left', legend_bbox=(1, 0.5), bbox_transform=None,
                              show_impedance: bool = False):
        """Helper to draw bus type visualization on a specific axis."""
        print(f"Calculating layout '{layout_name}' for bus types...")
        # 1. Calculate Layout
        if layout_name == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph)
        elif layout_name == 'yifan_hu':
            pos = self._yifan_hu_layout(graph)
        elif layout_name == 'voltage_layered':
            pos = self._get_layered_layout(graph)
        elif layout_name == 'hierarchical_tree':
            pos = self._hierarchical_tree_layout(graph)
        elif layout_name == 'radial_tree':
            pos = self._radial_tree_layout(graph)
        else:
            pos = nx.spring_layout(graph, seed=42)

        ax.clear()

        # 2. Node Config
        node_styles = {
            'Gen':  {'color': '#d62728', 'shape': 'o', 'label': 'Generation (Gen)'}, # Red
            'Load': {'color': '#2ca02c', 'shape': '^', 'label': 'Load (Load)'},       # Green
            'Conn': {'color': '#1f77b4', 'shape': 's', 'label': 'Connection (Conn)'}  # Blue
        }
        
        for n_type, style in node_styles.items():
            nodelist = [n for n, d in graph.nodes(data=True) if d.get('bus_type') == n_type]
            if nodelist:
                nx.draw_networkx_nodes(graph, pos, 
                                     nodelist=nodelist, 
                                     node_color=style['color'], 
                                     node_shape=style['shape'], 
                                     node_size=60, 
                                     alpha=0.9, 
                                     ax=ax, 
                                     label=style['label'])

        # 3. Edge Config
        if show_impedance:
            # Overwrite edge styles with impedance colors
            edge_coll = self._draw_edges_impedance(ax, graph, pos)
            if edge_coll:
                # Horizontal Colorbar at Bottom
                plt.colorbar(edge_coll, ax=ax, label="Impedance (Z)", orientation='horizontal', fraction=0.046, pad=0.04)
        else:
            edge_styles = {
                frozenset(['Gen', 'Gen']):  {'style': 'dashed', 'color': 'black', 'label': 'GG (Gen-Gen)'},
                frozenset(['Load', 'Load']): {'style': 'solid',  'color': 'black', 'label': 'LL (Load-Load)'},
                frozenset(['Conn', 'Conn']): {'style': 'dotted', 'color': 'black', 'label': 'CC (Conn-Conn)'},
                frozenset(['Gen', 'Load']):  {'style': 'dashdot', 'color': 'gray', 'label': 'GL (Gen-Load)'},
                frozenset(['Gen', 'Conn']):  {'style': (0, (3, 5, 1, 5)), 'color': 'gray', 'label': 'GC (Gen-Conn)'},
                frozenset(['Load', 'Conn']): {'style': (0, (5, 10)), 'color': 'gray', 'label': 'LC (Load-Conn)'},
            }

            for u, v in graph.edges():
                t1 = graph.nodes[u].get('bus_type', 'Unknown')
                t2 = graph.nodes[v].get('bus_type', 'Unknown')
                
                pair = frozenset([t1, t2])
                style = edge_styles.get(pair, {'style': 'solid', 'color': 'lightgray'})
                
                nx.draw_networkx_edges(graph, pos, 
                                       edgelist=[(u, v)], 
                                       style=style['style'], 
                                       edge_color=style['color'], 
                                       alpha=0.6, 
                                       ax=ax)

        # 4. Legend
        handles = []
        for style in node_styles.values():
            # Use smaller fixed size for legend (markersize=8)
            handle = mlines.Line2D([], [], 
                                 color=style['color'], 
                                 marker=style['shape'], 
                                 linestyle='None', 
                                 markersize=8, 
                                 label=style['label'])
            handles.append(handle)
        
        if not show_impedance:
            # Only show edge style legend if we aren't using impedance coloring
            for pair_key, style in edge_styles.items():
                line = mlines.Line2D([], [], color=style['color'], linestyle=style['style'], label=style['label'])
                handles.append(line)

        # Apply specific legend location args
        kwargs = {'handles': handles, 'loc': legend_loc, 'title': "Grid Components"}
        
        # Only apply bbox and transform if explicitly provided (for sidebar usage)
        if legend_bbox is not None:
             kwargs['bbox_to_anchor'] = legend_bbox
        if bbox_transform is not None:
            kwargs['bbox_transform'] = bbox_transform
            
        ax.legend(**kwargs)
        
        ax.set_title(f"{title}\nLayout: {layout_name}")
        ax.axis('off')

    def plot_bus_types(self, graph: nx.Graph, layout: str = 'kamada_kawai', title: str = "Bus Type Visualization", 
                      show_impedance: bool = False, figsize: Tuple[int, int] = (12, 10)):
        """Visualizes the grid coloring nodes by their Bus Type (Static). Option to show impedance on edges."""
        plt.figure(figsize=figsize)
        # Use 'best' location for adaptive placement in static plot
        self._draw_bus_types_on_ax(plt.gca(), graph, layout, title, legend_loc='best', legend_bbox=None, show_impedance=show_impedance)
        plt.tight_layout()
        plt.show()

    def plot_interactive_bus_types(self, graph: nx.Graph, title: str = "Interactive Bus Type Visualization", figsize: Tuple[int, int] = (14, 10)):
        """Opens an interactive window for Bus Type Visualization with layout selection."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create sidebar on the left
        plt.subplots_adjust(left=0.25)
        
        # Dropdown in sidebar (Top Left)
        dropdown_rect = [0.02, 0.85, 0.20, 0.05] 
        layout_options = ['kamada_kawai', 'yifan_hu', 'spring', 'spectral', 'voltage_layered', 'hierarchical_tree', 'radial_tree']
        
        def update_layout(label):
            # Legend below dropdown in sidebar
            self._draw_bus_types_on_ax(ax, graph, label, title,
                                     legend_loc='upper left',
                                     legend_bbox=(0.02, 0.8), # Below dropdown (0.85)
                                     bbox_transform=fig.transFigure)
            fig.canvas.draw_idle()

        dropdown = MatplotlibDropdown(fig, dropdown_rect, layout_options, active=0, on_select=update_layout)
        fig.text(0.02, 0.91, "Select Layout:", weight='bold')

        self._widgets.append(dropdown)
        
        # Initial draw with first option
        update_layout(layout_options[0])
        plt.show()

    def _draw_graph_on_ax(self, ax, graph, layout_name, title, show_labels,
                          legend_loc='upper right', legend_bbox=None, bbox_transform=None,
                          show_impedance: bool = False):
        """Helper to draw graph on a specific axis."""
        print(f"Calculating layout '{layout_name}'...")
        if layout_name == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph)
        elif layout_name == 'spring':
            pos = nx.spring_layout(graph, seed=42)
        elif layout_name == 'spectral':
            pos = nx.spectral_layout(graph)
        elif layout_name == 'voltage_layered':
            pos = self._get_layered_layout(graph)
        elif layout_name == 'yifan_hu':
            pos = self._yifan_hu_layout(graph)
        elif layout_name == 'hierarchical_tree':
            pos = self._hierarchical_tree_layout(graph)
        elif layout_name == 'radial_tree':
            pos = self._radial_tree_layout(graph)
        else:
            pos = nx.spring_layout(graph)

        node_colors = self._get_node_colors(graph)
        
        ax.clear()
        
        if show_impedance:
             edge_coll = self._draw_edges_impedance(ax, graph, pos)
             if edge_coll:
                # Horizontal Colorbar at Bottom
                plt.colorbar(edge_coll, ax=ax, label="Impedance (Z)", orientation='horizontal', fraction=0.046, pad=0.04)
        else:
            nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.3, edge_color='gray')
            
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=50, alpha=0.9)
        
        if show_labels:
            nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
            
        ax.set_title(f"{title}\nLayout: {layout_name}")
        ax.axis('off')

        # Add legend
        unique_levels = sorted(list(set(nx.get_node_attributes(graph, 'voltage_level').values())))
        import matplotlib.patches as mpatches
        legend_elements = [mpatches.Patch(color=self.cmap(lvl), label=f'Voltage Level {lvl}') for lvl in unique_levels]
        
        kwargs = {'handles': legend_elements, 'loc': legend_loc}
        if legend_bbox: kwargs['bbox_to_anchor'] = legend_bbox
        if bbox_transform: kwargs['bbox_transform'] = bbox_transform
            
        ax.legend(**kwargs)

    def _create_interactive_window(self, graph: nx.Graph, title: str, figsize: Tuple[int, int]):
        """Helper to create a figure with a Dropdown menu for layout selection."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create sidebar on the left
        plt.subplots_adjust(left=0.25)
        
        # Define Dropdown Position (Top Left corner)
        dropdown_rect = [0.02, 0.85, 0.20, 0.05] 
        
        layout_options = ['kamada_kawai', 'yifan_hu', 'spring', 'spectral', 'voltage_layered', 'hierarchical_tree', 'radial_tree']
        
        # Callback wrapper
        def update_layout(label):
            # Legend below dropdown
            self._draw_graph_on_ax(ax, graph, label, title, show_labels=False,
                                   legend_loc='upper left',
                                   legend_bbox=(0.02, 0.8),
                                   bbox_transform=fig.transFigure)
            fig.canvas.draw_idle()

        # Create our custom dropdown
        dropdown = MatplotlibDropdown(fig, dropdown_rect, layout_options, active=0, on_select=update_layout)
        
        # Label next to dropdown
        fig.text(0.02, 0.91, "Select Layout:", weight='bold')

        # Keep reference to prevent GC
        self._widgets.append(dropdown)
        
        # Initial draw
        update_layout(layout_options[0])
        plt.show()

    def plot_interactive(self, graph: nx.Graph, title: str = "Interactive Grid", figsize: Tuple[int, int] = (14, 10)):
        """Opens an interactive window for the full grid."""
        self._create_interactive_window(graph, title, figsize)

    def plot_interactive_voltage_level(self, graph: nx.Graph, level: int, title: Optional[str] = None, figsize: Tuple[int, int] = (12, 10)):
        """Opens an interactive window for a specific voltage level."""
        nodes = [n for n, d in graph.nodes(data=True) if d.get('voltage_level') == level]
        if not nodes:
            print(f"No nodes found for voltage level {level}")
            return
            
        subgraph = graph.subgraph(nodes)
        if title is None:
            title = f"Voltage Level {level} (Interactive)"
        
        self._create_interactive_window(subgraph, title, figsize)

    def plot_impedance(self, grid: nx.Graph, layout: str = 'kamada_kawai', title: str = "Transmission Line Impedance", figsize: Tuple[int, int] = (12, 10)):
        """
        Plots the grid with edges colored by their impedance magnitude (Z).
        Blue = Low Impedance (Strong), Red = High Impedance (Weak).
        """
        plt.figure(figsize=figsize)
        ax = plt.gca()

        # 1. Layout
        if layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(grid)
        elif layout == 'yifan_hu':
            pos = self._yifan_hu_layout(grid)
        elif layout == 'spring':
             pos = nx.spring_layout(grid, seed=42)
        elif layout == 'voltage_layered':
            pos = self._get_layered_layout(grid)
        elif layout == 'hierarchical_tree':
            pos = self._hierarchical_tree_layout(grid)
        elif layout == 'radial_tree':
            pos = self._radial_tree_layout(grid)
        else:
             pos = nx.kamada_kawai_layout(grid)

        # 2. Draw Nodes
        nx.draw_networkx_nodes(grid, pos, node_size=20, node_color='black', ax=ax)

        # 3. Draw Edges with Colormap using Helper
        edge_coll = self._draw_edges_impedance(ax, grid, pos)
        if edge_coll:
            # Horizontal Colorbar at Bottom
            plt.colorbar(edge_coll, ax=ax, label="Impedance Magnitude (Z) [p.u.]", orientation='horizontal', fraction=0.046, pad=0.04)
        
        ax.axis('off')
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        plt.show()