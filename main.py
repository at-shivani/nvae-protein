import monolith
import readutils
import networkx as nx 
import numpy as np 
import os 
import json 
from sklearn.model_selection import train_test_split 

NETWORK_FILE = '/Users/v.pathak/Projects/shivani_project/raw_data/mashup/data/networks/human/human_string_database_adjacency.txt'
NET_GENES_FILE = "/Users/v.pathak/Projects/shivani_project/raw_data/mashup/data/networks/human/human_string_genes.txt"

ANNOT_FILE = "/Users/v.pathak/Projects/shivani_project/raw_data/mashup/data/annotations/human/go_human_ref_bp_adjacency.txt"
ANNOT_GENES_FILE = "/Users/v.pathak/Projects/shivani_project/raw_data/mashup/data/annotations/human/go_human_ref_genes.txt"
LABELS_NAMES_FILE = "/Users/v.pathak/Projects/shivani_project/raw_data/mashup/data/annotations/human/go_human_ref_bp_terms.txt"
SUBSET_LABELS_FILE = "/Users/v.pathak/Projects/shivani_project/nvae-protein/only_keep_func.txt"
FOLDER = "/Users/v.pathak/Projects/shivani_project/nvae-protein/output"

def add_folder(f):
    def fn(obj, filename):
        path = os.path.join(FOLDER, filename)
        f(obj, path)
        print(f'Saved Successfully at: {path}')
    return fn 

def add_folder_open(f):
    def fn(filename):
        path = os.path.join(FOLDER, filename)
        return f(path)
    return fn 


@add_folder
def save_json(obj, name):
    name = f'{name}.json'
    with open(name, 'w') as f:
        json.dump(obj, f)

@add_folder
def save_np(obj, name):
    name = f'{name}.npy'
    assert isinstance(obj, np.ndarray)
    np.save(name, obj)

@add_folder
def save_txt(obj, name):
    obj = str(obj)
    name = f'{name}.txt'
    with open(name, 'w') as f:
        f.write(obj)

@add_folder_open
def load_json(name):
    name = f'{name}.json'
    with open(name, 'r') as f:
        return json.load(f)

@add_folder_open 
def load_np(name):
    name = f"{name}.npy"
    return np.load(name)

@add_folder_open
def load_txt(name):
    name = f'{name}.txt'
    return readutils.readfoldertext(name)


def main(all_nodes=False, cached=False):
    if cached:
        labels = load_json('labels')
        node_list = load_json('node_list')
        network_feats = load_np('network_feats')
        labels_one_hot = load_np('labels_one_hot')

    else:
        # reading networks
        graph = monolith.read_newtork_from_edgelist_txt(NETWORK_FILE)
        labels = monolith.read_labels(ANNOT_FILE, NET_GENES_FILE, LABELS_NAMES_FILE, SUBSET_LABELS_FILE)

        # selected_nodes = graphutils.n_hop_sample(graph, k=5, h=2)
        selected_nodes = list(map(int, monolith.readutils.readtextfile("/Users/v.pathak/Projects/shivani_project/nvae-protein/sample_nodes.txt")))
        subgraph = nx.subgraph(graph, selected_nodes) if not all_nodes else graph


        print(f'learning feats for {len(subgraph)} nodes')
        network_feats = monolith.learn_node_features(subgraph)
        labels_one_hot = monolith.prepare_dataset(subgraph, labels, NET_GENES_FILE, as_one_hot=True)
        node_list = [n for n in subgraph.nodes()]

        save_json(labels, 'labels')
        save_json(node_list, 'node_list')
        save_np(network_feats, 'network_feats')
        save_np(labels_one_hot, 'labels_one_hot')
    
    X, y = network_feats, labels_one_hot
    xtrain, xvalid, ytrain, yvalid = train_test_split(X, y)





    


    
    




    


    



    pass 


if __name__ == '__main__':
    main(all_nodes=True)