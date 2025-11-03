"""
Plotting file containing the functions to plot the proposed location/allocation decisions given by the model.
Copyright (c) 2025 Fair Isaac Corporation
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

# Packages imported from pip
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

def draw_sol(folder:str, instance, built:dict, E:dict, iteration: int=None):
    """
    Given a solution, draw the resulting map.

    Args:
        folder (str): Location to store the png file with the graph drawing.
        instance (FixedInstance): The container with all the input data for which we are solving.
        built (dict): The dictionary of camp locations installed.
        E (dict): The dictionary containing the allocation of school demand to soccer camp location
        iteration (int): The current iteration count number. Used to keep track in the filename.

    Returns:
        None
    """

    if not iteration:
        iteration=0

    mpl.rcParams['figure.figsize'] = (7,7)  # To plot with the right aspect ratio
    
    V = instance.schools + instance.sites  # Set of nodes

    # Set of edges: condition i<j implies these are EDGES, not arcs, 
    # and therefore they are not directed

    # Create a dictionary with nodes as keys and (x,y) tuples as their values
    coordS = {i:     tuple(instance.coord_schools[i]) for i in instance.schools}
    coordA = {j: tuple(instance.coord_sites[j])   for j in instance.sites}

    coord = {**coordS, **coordA}

    node_colS  = {i: '#5555ff' for i in instance.schools}
    node_colA1 = {j: '#ff5555' for j in instance.sites if built[j] > 0.5}
    node_colA0 = {j: "#a0a0a0" for j in instance.sites if built[j] < 0.5}
    node_col = {**node_colS, **node_colA1, **node_colA0}
    node_col = [node_col[i] for i in V]

    g = nx.Graph()

    g.add_nodes_from(V)
    g.add_edges_from(E)
    

    # Offset the labels
    offset = {node: (coord[node][0] + 0.2, coord[node][1] + 0.2) for node in g.nodes()}
    nx.draw_networkx_labels(g, pos=offset)
    
    nx.draw_networkx_edge_labels(g, pos=offset, edge_labels=E, font_color='#5555ff')

    nx.draw_networkx(g, pos=coord, node_color=node_col, node_shape='.', with_labels=False)
    

    # plt.show(block=False)
    plt.savefig(os.path.join(folder, f"solution_{iteration}.png"), format="PNG") 
    plt.close()