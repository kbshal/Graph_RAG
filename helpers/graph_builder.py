import networkx as nx
import matplotlib.pyplot as plt

class ConceptGraph:
    def __init__(self, data_frame: str = None):
        self.data_frame = data_frame
    
    def build_graph(self):
        self.graph = nx.from_pandas_edgelist(self.data_frame, 'node_1', 'node_2', edge_attr='edge', create_using=nx.MultiGraph())
        nx.draw(self.graph, with_labels=True)
    
    def extract_subgraph(self, central_node):
        connected_nodes = list(self.graph.neighbors(central_node)) + [central_node]
        subgraph = self.graph.subgraph(connected_nodes)

        layout = nx.spring_layout(subgraph)

        plt.figure(figsize=(8, 8))

        node_size = 2000
        node_color = 'lightblue'
        font_color = 'black'
        font_weight = 'bold'
        font_size = 8
        edge_color = 'gray'
        edge_style = 'dashed'

        nx.draw(subgraph, layout, with_labels=True, node_size=node_size, node_color=node_color, font_color=font_color, font_size=font_size,
                font_weight=font_weight, edge_color=edge_color, style=edge_style)

        plt.title(f"Subgraph of Node: {central_node}")

        plt.savefig('extracted_subgraph.png')
