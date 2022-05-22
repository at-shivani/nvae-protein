
import random 
import networkx

def get_n_hop_neighbors(graph, snode, n):
    all_nodes = set()
    all_nodes.add(snode) 
    for _ in range(n):
        new_nodes = set() 
        for n in all_nodes:
            new_nodes.update({ngh for ngh in graph.neighbors(n)})
        all_nodes.update(new_nodes)
    return list(all_nodes)
        

def n_hop_sample(g, h=2, k=2):
    all_nodes = list(g.nodes())
    starting_nodes = random.choices(all_nodes, k=k)
    nnodes = set()
    for snode in starting_nodes:
        nnodes.update({n for n in get_n_hop_neighbors(graph, snode, h)})
    return list(nnodes)