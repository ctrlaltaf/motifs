import argparse
import csv
import networkx as nx
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time
from functools import lru_cache
import os
from matplotlib import pyplot as plt
num_cores = os.cpu_count()



def draw_labeled_multigraph(G, attr_name, ax=None):
    """
    Length of connectionstyle must be at least that of a maximum number of edges
    between pair of nodes. This number is maximum one-sided connections
    for directed graph and maximum total connections for undirected graph.
    """

    pos = nx.shell_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=20, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="grey", ax=ax)

    labels = {
        tuple(edge): f"{attr_name}={attrs[attr_name]}"
        for *edge, attrs in G.edges(keys=True, data=True)
    }
    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        label_pos=0.3,
        font_color="blue",
        bbox={"alpha": 0},
        ax=ax,
    )

def process_edges(
    file_path, G, protein_id_dict, visited_nodes, label, node_limit, edge_limit
):
    """Helper function to process edges and add them to the graph."""
    print(f"Currently processing {label} edges")
    with open(file_path, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader) 
        node_count = len(visited_nodes)
        edge_count = G.number_of_edges()

        for row in csv_reader:
            if node_count > node_limit or edge_count > edge_limit:
                break

            id1 = row[0]
            id2 = row[1]

            # Map protein IDs to unique integers if not already mapped
            if id1 not in protein_id_dict:
                protein_id_dict[id1] = len(protein_id_dict) + 1

            if id2 not in protein_id_dict:
                protein_id_dict[id2] = len(protein_id_dict) + 1

            mapped_id1 = protein_id_dict[id1]
            mapped_id2 = protein_id_dict[id2]

            if mapped_id1 not in visited_nodes:
                visited_nodes.add(mapped_id1)
                node_count += 1

            if mapped_id2 not in visited_nodes:
                visited_nodes.add(mapped_id2)
                node_count += 1

            G.add_edge(mapped_id1, mapped_id2, label=label)
            edge_count += 1


def read_csv(
    ppi_path,
    reg_path,
    node_size_limit=float("inf"),
    edge_size_limit=float("inf"),
):
    """Reads CSV files and constructs a graph with edges labeled as 'ppi' or 'reg'."""
    G = nx.MultiDiGraph()
    visited_nodes = set()
    protein_id_dict = {}

    process_edges(
        ppi_path,
        G,
        protein_id_dict,
        visited_nodes,
        "ppi",
        node_size_limit,
        edge_size_limit,
    )
    process_edges(
        reg_path,
        G,
        protein_id_dict,
        visited_nodes,
        "reg",
        node_size_limit,
        edge_size_limit,
    )

    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")
    print()

    return G, protein_id_dict

def get_three_node_graphlet_dist_adj_list_v2(G: nx.MultiDiGraph, G_prime: nx.Graph):
    three_node_graphlet_dict = {}
    graphlet_mapper = {}
    start_time = time.time()

    # Create adjacency matrix with labels
    node_list = list(G.nodes())
    node_index = {node: idx for idx, node in enumerate(node_list)}
    adj_matrix = np.zeros((len(node_list), len(node_list), 3), dtype=int)

    for i, j, data in G.edges(data=True):
        label = data.get("label")
        idx_i = node_index[i]
        idx_j = node_index[j]
        if label == "ppi":
            adj_matrix[idx_i, idx_j, 0] = 1
            adj_matrix[idx_j, idx_i, 0] = 1
        elif label == "reg":
            adj_matrix[idx_i, idx_j, 1] += 1
            adj_matrix[idx_j, idx_i, 2] += 1

    sorted_nodes = sorted(G_prime.nodes(), key=lambda node: G_prime.degree(node))
    # for node in sorted_nodes:
    #     print(f"{node} : {G.degree(node)}")
    @lru_cache(None)
    def get_combinations(nodes):
        return list(combinations(nodes, 2))

    def process_node(i):
        i_neighbors = set(G_prime.neighbors(i))
        local_graphlet_dict = {}

        # Convert the set to a sorted tuple
        sorted_neighbors = tuple(sorted(i_neighbors))
        # print(sorted_neighbors)
        hashed = False
        for j, k in get_combinations(sorted_neighbors):

            idx_i, idx_j, idx_k = node_index[i], node_index[j], node_index[k]

            ab = adj_matrix[idx_i, idx_j]
            ac = adj_matrix[idx_i, idx_k]
            ba = adj_matrix[idx_j, idx_i]
            bc = adj_matrix[idx_j, idx_k]
            ca = adj_matrix[idx_k, idx_i]
            cb = adj_matrix[idx_k, idx_j]

            a_b, a_c, b_a, b_c, c_a, c_b = (
                hash(tuple(ab)),
                hash(tuple(ac)),
                hash(tuple(ba)),
                hash(tuple(bc)),
                hash(tuple(ca)),
                hash(tuple(cb)),
            )

            a_edges = tuple(sorted([a_b, a_c]))
            b_edges = tuple(sorted([b_a, b_c]))
            c_edges = tuple(sorted([c_a, c_b]))

            sorted_tuples = tuple(
                sorted([a_edges, b_edges, c_edges], key=lambda x: (x[0], x[1]))
            )

            hash_val = hash(sorted_tuples)
            if hash_val not in local_graphlet_dict:
                local_graphlet_dict[hash_val] = 0
                graphlet_mapper[hash_val] = sorted_tuples
                hashed = True
            local_graphlet_dict[hash_val] += 1

            # print(completed_i)
        # print(f"{i} : {local_graphlet_dict}")
        return local_graphlet_dict  # Return the local graphlet dictionary
    
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        local_graphlet_dicts = list(executor.map(process_node, sorted_nodes, chunksize=10))

    # Merge the local graphlet dictionaries into the global one
    for local_dict in local_graphlet_dicts:
        for graphlet_hash, count in local_dict.items():
            if graphlet_hash not in three_node_graphlet_dict:
                three_node_graphlet_dict[graphlet_hash] = 0
            three_node_graphlet_dict[graphlet_hash] += count

    run_time = time.time() - start_time
    print(f"run time : {run_time:.3f} seconds")

    return three_node_graphlet_dict, graphlet_mapper


def main(ppi_path, reg_path, node_limit, edge_limit):
    """Main function to handle graph construction."""
    G, protein_id_dict = read_csv(ppi_path, reg_path, node_limit, edge_limit)

    # Create an undirected version of the graph for graphlet analysis
    G_prime = nx.Graph()
    for u, v in G.edges():
        if not G_prime.has_edge(u, v):
            G_prime.add_edge(u, v)

    # Calculate graphlet distribution using ThreadPoolExecutor
    graphlet_dict, graphlet_mapper = get_three_node_graphlet_dist_adj_list_v2(G, G_prime)

    # Print results
    print("Three-node graphlet distribution:")
    for graphlet_hash, count in graphlet_dict.items():
        print(f"Graphlet: {graphlet_mapper[graphlet_hash]}, Count: {count}")
    print(f"unique graphlets found : {len(graphlet_dict)}")

    draw_labeled_multigraph(G, "label")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graphlet Analysis Tool")
    parser.add_argument("ppi_path", type=str, help="Path to the PPI CSV file")
    parser.add_argument("reg_path", type=str, help="Path to the regulatory CSV file")
    parser.add_argument("node_limit", type=int, help="Maximum number of nodes in the graph")
    parser.add_argument("edge_limit", type=int, help="Maximum number of edges in the graph")

    args = parser.parse_args()

    main(
        args.ppi_path,
        args.reg_path,
        args.node_limit,
        args.edge_limit,
    )