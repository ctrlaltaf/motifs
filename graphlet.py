import sys
from matplotlib import pyplot as plt
from pathlib import Path
import networkx as nx
import numpy
import random
import scipy as sp
import csv
import curses


def get_two_node_dict():
    """
    Get all 2-node graphlet binary edge vectors
    see diagram for reference
    """
    # 2-node graphlet binary edge vectors
    # see diagram for reference
    a_1 = (1, 0, 0)
    a_2 = (1, 0, 0)
    a_hash = hash(a_1 + a_2)

    b_1 = (0, 1, 0)
    b_2 = (0, 0, 1)
    b_hash = hash(b_1 + b_2)

    c_1 = (1, 1, 0)
    c_2 = (1, 0, 1)
    c_hash = hash(c_1 + c_2)

    d_1 = (1, 1, 1)
    d_2 = (1, 1, 1)
    d_hash = hash(d_1 + d_2)

    e_1 = (0, 1, 1)
    e_2 = (0, 1, 1)
    e_hash = hash(e_1 + e_2)

    two_node_graphlet_dict = {a_hash: 0, b_hash: 0, c_hash: 0, d_hash: 0, e_hash: 0}
    two_node_graphlet_labels = {a_hash: 1, b_hash: 2, c_hash: 3, d_hash: 4, e_hash: 5}
    return two_node_graphlet_dict, two_node_graphlet_labels


def generate_random_multi_graph(n, edge_probability=0.3, edge_label_probability=0.3):
    """
    Given the number of nodes ,n, and some edge and label probabilites,
    generate a random MultiDiGraph
    """

    print("Generating random graph")
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(random.randrange(0, 50)):
            if i != j:
                if random.random() < edge_probability:
                    if random.random() < edge_label_probability:
                        G.add_edge(i, j, label="ppi")
                    else:
                        G.add_edge(i, j, label="reg")
                        if random.random() < 0.7:
                            G.add_edge(j, i, label="reg")
                # want to generate a graph that is connected
                elif G.degree(j) == 0:
                    if random.random() < edge_label_probability:
                        G.add_edge(j, i, label="ppi")
                    else:
                        G.add_edge(j, i, label="reg")
    return G


def update_protein_id_dict(file_path, res_dict, start_index):
    """Helper function to update the protein ID dictionary from a file."""
    with open(file_path, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            for protein_id in row[:2]:
                res_dict.setdefault(protein_id, start_index)
                if res_dict[protein_id] == start_index:
                    start_index += 1
    return start_index


def get_protein_id_dict(ppi_path, reg_path):
    """Creates a dictionary mapping protein IDs to unique integers."""
    res_dict = {}
    start_index = 0
    start_index = update_protein_id_dict(ppi_path, res_dict, start_index)
    update_protein_id_dict(reg_path, res_dict, start_index)
    return res_dict


def process_edges(
    file_path, G, protein_id_dict, visited_nodes, label, node_limit, edge_limit
):
    """Helper function to process edges and add them to the graph."""
    print(f"currently processing {label} edges")
    with open(file_path, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        node_count = len(visited_nodes)
        edge_count = G.number_of_edges()

        for row in csv_reader:
            if node_count > node_limit or edge_count > edge_limit:
                break

            id1 = protein_id_dict[row[0]]
            id2 = protein_id_dict[row[1]]

            if id1 not in visited_nodes:
                visited_nodes.add(id1)
                node_count += 1

            if id2 not in visited_nodes:
                visited_nodes.add(id2)
                node_count += 1

            G.add_edge(id1, id2, label=label)
            edge_count += 1


def read_csv(
    ppi_path,
    reg_path,
    protein_id_dict,
    node_size_limit=float("inf"),
    edge_size_limit=float("inf"),
):
    """Reads CSV files and constructs a graph with edges labeled as 'ppi' or 'reg'."""
    G = nx.MultiDiGraph()
    visited_nodes = set()

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
    print()

    return G


def draw_labeled_multigraph(G, attr_name, ax=None):
    """
    Length of connectionstyle must be at least that of a maximum number of edges
    between pair of nodes. This number is maximum one-sided connections
    for directed graph and maximum total connections for undirected graph.
    """

    pos = nx.spring_layout(G)
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


def get_two_node_graphlet_stats(G, two_node_graphlet_dict):
    G_adj_list = get_two_node_adjacency_list(G)
    orbits_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
    for neighbors in G_adj_list:
        i = neighbors[0]
        # print((i / len(G.nodes())) * 100, end="\r")
        for j in neighbors[1]:
            vectors = G_adj_list[i][1][j] + G_adj_list[j][1][i]
            orbits_dict = get_two_node_orbit_position(
                i, j, G_adj_list[i][1][j], G_adj_list[j][1][i], orbits_dict
            )
            if hash(tuple(vectors)) in two_node_graphlet_dict:
                two_node_graphlet_dict[hash(tuple(vectors))] += 1
    return two_node_graphlet_dict, orbits_dict


def get_two_node_adjacency_list(G):
    """Get the adjacency list for a MultiDiGraph"""
    print("getting adjacency list")

    adj_list_vector = [{} for _ in range(len(G.nodes()))]

    for i, j, data in G.edges(data=True):
        label = data.get("label")
        if label == "ppi":
            if j not in adj_list_vector[i]:
                adj_list_vector[i][j] = [1, 0, 0]
            # else:
            #     adj_list_vector[i][j][0] += 1
            if i not in adj_list_vector[j]:
                adj_list_vector[j][i] = [1, 0, 0]
            # else:
            #     adj_list_vector[j][i][0] += 1
        elif label == "reg":
            if j not in adj_list_vector[i]:
                adj_list_vector[i][j] = [0, 1, 0]
            else:
                adj_list_vector[i][j][1] += 1
            if i not in adj_list_vector[j]:
                adj_list_vector[j][i] = [0, 0, 1]
            else:
                adj_list_vector[j][i][2] += 1

    final_adj_list_vector = [
        (i, neighbors) for i, neighbors in enumerate(adj_list_vector)
    ]
    return final_adj_list_vector


def get_two_node_orbit_position(a, b, a_vec, b_vec, orbit_dict):
    # check orbit positioning via vectors
    if a_vec == [1, 0, 0] and b_vec == [1, 0, 0]:
        if a not in orbit_dict[1]:
            orbit_dict[1] += [a]
        if b not in orbit_dict[1]:
            orbit_dict[1] += [b]
    elif (
        a_vec == [0, 1, 0]
        and b_vec == [0, 0, 1]
        or a_vec == [0, 0, 1]
        and b_vec == [0, 1, 0]
    ):
        if a_vec == [0, 1, 0] and a not in orbit_dict[2]:
            orbit_dict[2] += [a]
            orbit_dict[3] += [b]
        elif b not in orbit_dict[2]:
            orbit_dict[2] += [b]
            orbit_dict[3] += [a]

    elif (
        a_vec == [1, 1, 0]
        and b_vec == [1, 0, 1]
        or a_vec == [1, 0, 1]
        and b_vec == [1, 1, 0]
    ):
        if a_vec == [1, 1, 0] and a not in orbit_dict[4]:
            orbit_dict[4] += [a]
            orbit_dict[5] += [b]
        elif b not in orbit_dict[4]:
            orbit_dict[4] += [b]
            orbit_dict[5] += [a]

    elif a_vec == [1, 1, 1] and b_vec == [1, 1, 1]:
        if a not in orbit_dict[6]:
            orbit_dict[6] += [a]
        if b not in orbit_dict[6]:
            orbit_dict[6] += [b]

    elif a_vec == [0, 1, 1] and b_vec == [0, 1, 1]:
        if a not in orbit_dict[7]:
            orbit_dict[7] += [a]
        if b not in orbit_dict[6]:
            orbit_dict[7] += [b]

    return orbit_dict


def get_two_node_graphlet_dist_adj_matrix(G, two_node_graphlet_dict):
    G_adj_matrix = get_adjacency_matrix(G)
    for i in range(len(G_adj_matrix)):
        for j in range(len(G_adj_matrix[0])):
            vector = G_adj_matrix[i][j] + G_adj_matrix[j][i]
            if hash(tuple(vector)) in two_node_graphlet_dict:
                two_node_graphlet_dict[hash(tuple(vector))] += 1
    return two_node_graphlet_dict


def get_adjacency_matrix(G):
    """Get the adjacency matrix for a MultiDiGraph"""

    G_adj_matrix = nx.adjacency_matrix(G)
    adj_matrix = [
        [[0, 0, 0] for _ in range(len(G.nodes()))] for _ in range(len(G.nodes()))
    ]

    for edge in G.edges(data=True):
        i = edge[0]
        j = edge[1]
        edge_type = edge[2]["label"]

        if edge_type == "ppi":
            adj_matrix[i][j][0] = 1
            adj_matrix[j][i][0] = 1
        elif edge_type == "reg":
            adj_matrix[i][j][1] = 1
            adj_matrix[j][i][2] = 1

    return G_adj_matrix


def get_three_node_graphlet_dist_adj_list(G: nx.MultiDiGraph):
    three_node_graphlet_dict = {}

    # create all the binary edge vectors
    adj_list_vector = [{} for _ in range(len(G.nodes()))]

    for i, j, data in G.edges(data=True):
        label = data.get("label")
        if label == "ppi":
            if j not in adj_list_vector[i]:
                adj_list_vector[i][j] = [1, 0, 0]
            # else:
            #     adj_list_vector[i][j][0] += 1
            if i not in adj_list_vector[j]:
                adj_list_vector[j][i] = [1, 0, 0]
            # else:
            #     adj_list_vector[j][i][0] += 1
        elif label == "reg":
            if j not in adj_list_vector[i]:
                adj_list_vector[i][j] = [0, 1, 0]
            else:
                adj_list_vector[i][j][1] += 1
            if i not in adj_list_vector[j]:
                adj_list_vector[j][i] = [0, 0, 1]
            else:
                adj_list_vector[j][i][2] += 1

    # find all combinations of potential 3 node graphlets
    # pick an edge between A and B
    # for each edge pair, find the union of neighbors between A and B
    e_v = (0, 0, 0) # 0
    ppi_v = (1, 0, 0) # 1
    reg1_v = (0, 1, 0) # 2
    reg2_v = (0, 0, 1) # 4
    ppi_reg1_v = (1, 1, 0) # 5
    ppi_reg2_v = (1, 0, 1) # 6
    both_v = (1, 1, 1) # 7
    reg_v = (0, 1, 1) # 8
    vector_identifier_dict ={
        hash(e_v) : 0,
        hash(ppi_v) : 2,
        hash(reg1_v) : 3,
        hash(reg2_v) : 4,
        hash(ppi_reg1_v) : 5,
        hash(ppi_reg2_v) : 6,
        hash(both_v) : 7,
        hash(reg_v) : 8,
    }
    three_node_combination = []
    graphlet_groups = []
    max_reg = 0
    count = 0
    for (
        i,
        j,
    ) in G.edges():
        # print()
        neighbors = list()
        for node in (i, j):
            in_edges = list(G.in_edges(node))
            if in_edges:
                for edge in in_edges:
                    if edge[1] not in neighbors:
                        neighbors.append(edge[1])
            out_edges = list(G.out_edges(node))
            if out_edges:
                for edge in out_edges:
                    if edge[1] not in neighbors:
                        neighbors.append(edge[1])
        for k in neighbors:
            if k != i and k != j:
                three_node_combination.append([i, j, k])
                # for each triplet combination, we have to get all their binary edge vectors
                # for nodes A, B, C, there are six binary edge vectors:
                # A-B
                # A-C
                # B-A
                # B-C
                # C-A
                # C-B

                # i < j < k
                combination = sorted([i, j, k])
                # print(combination[0], combination[1], combination[2])
                a = combination[0]
                b = combination[1]
                c = combination[2]

                ab = ac = ba = bc = ca = cb = 0
                if b in adj_list_vector[a]:
                    ab = adj_list_vector[a][b]
                else:
                    ab = [0, 0, 0]
                if c in adj_list_vector[a]:
                    ac = adj_list_vector[a][c]
                else:
                    ac = [0, 0, 0]
                if a in adj_list_vector[b]:
                    ba = adj_list_vector[b][a]
                else:
                    ba = [0, 0, 0]
                if c in adj_list_vector[b]:
                    bc = adj_list_vector[b][c]
                else:
                    bc = [0, 0, 0]
                if a in adj_list_vector[c]:
                    ca = adj_list_vector[c][a]
                else:
                    ca = [0, 0, 0]
                if b in adj_list_vector[c]:
                    cb = adj_list_vector[c][b]
                else:
                    cb = [0, 0, 0]

                # reg_sum = (
                #     ab[1]
                #     + ab[2]
                #     + bc[1]
                #     + bc[2]
                #     + ca[1]
                #     + ca[2]
                #     + ab[0]
                #     + bc[0]
                #     + ca[0]
                # )
                # max_reg = max(reg_sum, max_reg)

                # vector = ab + ac + ba + bc + ca + cb
                # vector = [0,0,0]
                # for v in [ab, ac, ba, bc, ca, cb]:
                #     vector[0] += v[0]
                #     vector[1] += v[1]
                #     vector[2] += v[2]
                # # print(vector)
                # # [0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1 ,0,0,1]
                # if vector not in graphlet_groups:
                #     graphlet_groups += [vector]
                # # print(f"{i} {j} {k} = {vector_group}")
                # get_three_node_graphlet_dict(ab, ac, ba, bc, ca, cb)
                # if hash(tuple(vector)) not in three_node_graphlet_dict:
                #     three_node_graphlet_dict[hash(tuple(vector))] = 0
                # three_node_graphlet_dict[hash(tuple(vector))] += 1


                vector_id = []
                for v in [ab, ac, ba, bc, ca, cb]:
                    vector_id.append(vector_identifier_dict[hash(tuple(v))])
                # print(vector_id)
                vector_id = sorted(vector_id) 
                # vector_id = vector_id

                if vector_id not in graphlet_groups:
                    graphlet_groups.append(vector_id)
        count+=1
        print((count / len(G.edges())) * 100, end="\r")

    print(f"number of graphlet groups {len(graphlet_groups)}")
    # print()
    # for vector in graphlet_groups:
    #     print(vector)
    # print()
    print(f"max reg {max_reg}")

    return three_node_graphlet_dict


def get_three_node_graphlet_dict(ab, ac, ba, bc, ca, cb):
    e_v = [0, 0, 0]
    ppi_v = [1, 0, 0]
    reg1_v = [0, 1, 0]
    reg2_v = [0, 0, 1]
    ppi_reg1_v = [1, 1, 0]
    ppi_reg2_v = [1, 0, 1]
    both_v = [1, 1, 1]
    reg_v = [0, 1, 1]



    return None


def dropdown_menu(stdscr, options):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.curs_set(0)
    current_index = 0

    while True:
        stdscr.clear()

        for i, option in enumerate(options):
            x = 2
            y = 1 + i
            if i == current_index:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, option.ljust(20))
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, option.ljust(20))

        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_UP and current_index > 0:
            current_index -= 1
        elif key == curses.KEY_DOWN and current_index < len(options) - 1:
            current_index += 1
        elif key in [curses.KEY_ENTER, 10, 13]:
            return options[current_index]
        elif key == 27:
            return None


def type_answer(stdscr, step):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.curs_set(0)
    x = 0
    y = 1
    typed_number = ""

    while True:
        stdscr.clear()
        if step == 0:
            stdscr.addstr(0, 0, "Node limit? (Press ENTER for no limit)")
        elif step == 1:
            stdscr.addstr(0, 0, "Edge limit? (Press ENTER for no limit)")

        stdscr.addstr(y, x, typed_number)
        stdscr.refresh()

        key = stdscr.getkey()

        # Check if key is a digit (0-9)
        if key.isdigit():
            typed_number += key
        elif key in ("\n", "\r") and typed_number == "" or int(typed_number) == 0:
            return 999999999
        elif key in ("\n", "\r") and int(typed_number) != 0:
            return int(typed_number)


def get_user_inputs(selected_network, selected_graphlet):
    ppi_path = None
    reg_path = None
    graphlet_option = None

    match selected_network:
        case "D. melanogaster":
            ppi_path = Path("data/fly_ppi.csv")
            reg_path = Path("data/fly_reg.csv")
        case "B. subtilis":
            ppi_path = Path("data/bsub_ppi.csv")
            reg_path = Path("data/bsub_reg.csv")
        case "S. cerevisiae":
            ppi_path = Path("data/ceravisiae_ppi.csv")
            reg_path = Path("data/ceravisiae_reg.csv")
        case "D. rerio":
            ppi_path = Path("data/drerio_ppi.csv")
            reg_path = Path("data/drerio_reg.csv")
        case "C. elegans":
            ppi_path = Path("data/elegans_ppi.csv")
            reg_path = Path("data/elegans_reg.csv")
        case "Test network":
            ppi_path = Path("data/fly_ppi.csv")
            reg_path = Path("data/fly_reg.csv")

    if selected_graphlet == "2-node":
        graphlet_option = 2
    elif selected_graphlet == "3-node":
        graphlet_option = 3

    return ppi_path, reg_path, graphlet_option


def main(stdscr):
    try:
        network = [
            "D. melanogaster",
            "B. subtilis",
            "S. cerevisiae",
            "D. rerio",
            "C. elegans",
            "Test network",
            "Exit",
        ]
        selected_network = dropdown_menu(stdscr, network)
        stdscr.clear()

        graphlet = ["2-node", "3-node", "Exit"]
        selected_graphlet = dropdown_menu(stdscr, graphlet)
        stdscr.clear()

        node_limit = type_answer(stdscr, 0)
        stdscr.clear()

        edge_limit = type_answer(stdscr, 1)
        stdscr.clear()
    except Exception as e:
        stdscr.addstr(0, 0, f"An error occurred: {str(e)}")
        stdscr.getch()
    finally:
        curses.endwin()

    ppi_path, reg_path, graphlet_mode = get_user_inputs(
        selected_network, selected_graphlet
    )

    two_node_graphlet_dict, two_node_graphlet_labels = get_two_node_dict()
    protein_id_dict = get_protein_id_dict(ppi_path, reg_path)
    G = read_csv(
        ppi_path,
        reg_path,
        protein_id_dict,
        node_size_limit=node_limit,
        edge_size_limit=edge_limit,
    )
    print(selected_network)
    print(f"Number of nodes: {len(G.nodes())}")
    print(f"Number of edges: {len(G.edges())}")

    if graphlet_mode == 2:
        two_node_graphlet_dict, two_node_orbit_dict = get_two_node_graphlet_stats(
            G, two_node_graphlet_dict
        )
        print("\ntwo node graphlet counts")
        for key in two_node_graphlet_labels:
            print(f"G_{two_node_graphlet_labels[key]} = {two_node_graphlet_dict[key]}")
        print("\ntwo node graphlet orbit counts")
        for key in two_node_orbit_dict:
            print(f"{key} = {len(two_node_orbit_dict[key])}")
    elif graphlet_mode == 3:
        three_node_graphlet_dict = get_three_node_graphlet_dist_adj_list(G)
        print("\nthree node graphlet counts")
        for key in three_node_graphlet_dict:
            print(f"{key} = {three_node_graphlet_dict[key]}")
        print(f"number of 3 node graphlets = {len(three_node_graphlet_dict.keys())}")

    # draw_labeled_multigraph(G, "label")
    # plt.show()


if __name__ == "__main__":
    curses.wrapper(main)
