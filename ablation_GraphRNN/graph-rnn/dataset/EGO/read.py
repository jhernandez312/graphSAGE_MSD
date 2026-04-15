import pickle as pkl
import sys
import networkx as nx
import numpy as np

def load_file(name):
    with open(name, 'rb') as f:
        if sys.version_info > (3, 0):
            return pkl.load(f, encoding='latin1')
        else:
            return pkl.load(f)

x     = load_file("ind.citeseer.x")
tx    = load_file("ind.citeseer.tx")
allx  = load_file("ind.citeseer.allx")
graph = load_file("ind.citeseer.graph")

# test indices is plain text
test_idx = [int(i.strip()) for i in open("ind.citeseer.test.index.txt")]

print(type(x), x.shape)
print(type(graph), len(graph))

G = nx.from_dict_of_lists(graph)
A = nx.adjacency_matrix(G)

print(A.shape)