
import networkx as nx
import matplotlib.pyplot as plt

#### Build a graph ####

G = nx.Graph() #Initialise the graph

## Add nodes
G.add_node("A") #Add a single node
G.add_nodes_from(["B", "C", "D"]) #Add multiple nodes at once

## Add connections
G.add_edge("A", "B")
G.add_edges_from([("A", "C"), ("C", "D"), ("D", "A")])

#### Calculate metrics ####
G.degree() #Degree
nx.closeness_centrality(G) #Centrality
nx.betweenness_centrality(G) #Betweenness



