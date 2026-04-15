import networkx as nx
import pickle
import matplotlib.pyplot as plt


directed_dataset = []

# --- example graph 1: A Directed Triangle ---
g1 = nx.DiGraph()
g1.add_edges_from([(0, 1), (1, 2), (2, 0)])
directed_dataset.append(g1)

# to see graph:
nx.draw(g1, with_labels=True)
plt.show()

# --- example graph 2: A Star Graph with Outward Edges ---
g2 = nx.DiGraph()
g2.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])
directed_dataset.append(g2)

# --- example graph 3: A Chain with a Bidirectional Link ---
g3 = nx.DiGraph()
g3.add_edges_from([(0, 1), (1, 2), (2, 1), (2, 3)])
directed_dataset.append(g3)



# save list to pickle file. this saves the networkx graphs and the metadata so the "directed_dataset" list doesn't have to be rebuilt everytime. run only after graphs are completed

'''
with open("my_directed_graphs.pkl", "wb") as f:
    pickle.dump(directed_dataset, f)

print(f"Saved {len(directed_dataset)} directed graphs.")
'''