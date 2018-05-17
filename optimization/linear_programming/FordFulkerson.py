import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from networkx.drawing.nx_agraph import graphviz_layout

class FordFulkerson(object):
    """
    Described in https://en.wikipedia.org/wiki/Ford%E2%80%93Fulkerson_algorithm
    """
    def __init__(self, G, src, dest):
        self.G = G
        self.src = src
        self.dest = dest
        self.path = []

    def find_path(self):
        """
        Use dfs to find an augmented path.
        self.path is a list containing tuples of (first, second, weight)
        """

        node_stack = [self.src]
        prev_node = None
        visited_nodes = {}

        while len(node_stack) > 0:
            cur_node = node_stack.pop()
            visited_nodes[cur_node] = 1
            is_exhausted = True

            if prev_node is not None:
                self.path.append((prev_node, cur_node, G.get_edge_data(prev_node, cur_node)['weight']))

            # append neighbors
            neighbors = list(self.G.edges(cur_node, data=True))

            for nn in neighbors:
                if nn[1] == self.dest and nn[2]['weight'] > 0:
                    self.path.append((cur_node, nn[1], nn[2]['weight']))
                    return self.path
                else:
                    if nn[1] not in visited_nodes and nn[2]['weight'] > 0:
                        node_stack.append(nn[1])
                        is_exhausted = False

            if is_exhausted:
                p = self.path.pop()
                prev_node = p[0]
            else:
                prev_node = cur_node

        return None

    def update_residual_graph(self, flow_appended: int):
        """
        Update residual graph by constructing opposite flow
        """
        for p in self.path:
            u, v = p[0], p[1]
            G[u][v]['weight'] -= flow_appended

            if G.has_edge(v, u):
                G[v][u]['weight'] += flow_appended
            else:
                G.add_edge(v, u, weight=flow_appended)

    def get_flow_change(self) -> int:
        """
        The available flow is the minimum of edge
        """
        return min([p[2] for p in self.path])

    def compute_max_flow(self) -> int:
        cur_iter = 0
        max_iter = 100 # control iteration depth
        max_flow = 0

        while self.find_path() is not None and cur_iter < max_iter:
            flow_appended = self.get_flow_change()
            max_flow += flow_appended
            self.update_residual_graph(flow_appended)
            self.path = []
            cur_iter += 1

        return max_flow


if __name__ == '__main__':
    # --------------------------
    # Test case in https://www.geeksforgeeks.org/ford-fulkerson-algorithm-for-maximum-flow-problem/
    # --------------------------
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=16)
    G.add_edge(0, 2, weight=13)
    G.add_edge(1, 2, weight=10)
    G.add_edge(1, 3, weight=12)
    G.add_edge(2, 1, weight=4)
    G.add_edge(2, 4, weight=14)
    G.add_edge(3, 2, weight=9)
    G.add_edge(3, 5, weight=20)
    G.add_edge(4, 3, weight=7)
    G.add_edge(4, 5, weight=4)

    mdl = FordFulkerson(G, 0, 5)
    max_flow = mdl.compute_max_flow()

    # expect max flow to be 23
    print('computed max flow is %d and expected value should be 23' % max_flow)

    # --------------------------
    # Test case in https://www.cs.princeton.edu/courses/archive/spring13/cos423/lectures/07DemoFordFulkerson.pdf
    # --------------------------
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=10)
    G.add_edge(0, 2, weight=10)
    G.add_edge(1, 2, weight=2)
    G.add_edge(1, 3, weight=4)
    G.add_edge(1, 4, weight=8)
    G.add_edge(2, 4, weight=9)
    G.add_edge(3, 5, weight=10)
    G.add_edge(4, 5, weight=10)
    G.add_edge(4, 3, weight=6)

    mdl = FordFulkerson(G, 0, 5)
    max_flow = mdl.compute_max_flow()

    print('computed max flow is %d and expected value should be 19' % max_flow)

    plt.title('Ford Fulkerson Example')
    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.show()