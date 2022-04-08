"""
This module is used to check common parameters of the Graph and
to analyse the common characteristics of Graph.

networkx package is used for this module.

"""

AUTHOR = "Tanuja Seervi"

# Import the required libraries 

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import powerlaw


def find_adj_matrix(G):
    
    """
    This function returns the adjacency matrix for a given graph
    
    Input: networkx.classes.graph.Graph
    Output: scipy.sparse.csc.csc_matrix
    
    """
    
    # Construct an adjacency matrix
    a = nx.adjacency_matrix(G)

    # In the NetworkX package, for directed graphs, entry i,j
    # corresponds to an edge from i to j. Hence, we take the
    # transpose of the matrix to obtain the Adjacency matrix of
    # the graph as defined in the Newman textbook. 
    adj = np.transpose(a)

    return adj


def find_degrees(G, directed):

    """
    This function returns the degrees(in-degrees or out-degrees for 
    directed graph) for each node in a dictionary
    which is sorted on the keys(node) for a given input graph.
    
    Input: G: networkx.classes.graph.Graph
           directed: bool 
  
    Output: dict
            key = node, value = degree of the node
    
    """
    
    degree_dict = {}
    in_degree_dict = {}
    out_degree_dict = {}

    if directed:
        for n, d in sorted(dict(G.in_degree()).items()):
            in_degree_dict[n] = d

        for n, d in sorted(dict(G.out_degree()).items()):
            out_degree_dict[n] = d

        return in_degree_dict, out_degree_dict

    else:
        for n, d in sorted(dict(G.degree()).items()):
            degree_dict[n] = d

        return degree_dict


def find_eigenvector_centrality(G, i=100):
    
    """
    This function returns the eigenvector centrality for each node in
    a dictonary which is sorted on the keys(node) for a given graph
    
    Input: G: networkx.classes.graph.Graph
           i: int
                max_iter value for nx.eigenvector_centrality()
    Output: dict
            key = node, value = eigenvector centrality of the node
            (rounded to 3 decimal places)
    """
    
    eigenvector_cent = nx.eigenvector_centrality(G,max_iter=i)
    
    ev_cent_dict = {}
    for node, eigen_cent in sorted(eigenvector_cent.items()):
        ev_cent_dict[node] = round(eigen_cent,3)

    return ev_cent_dict


def find_katz_cent(G, i=100):
    
    """
    This function returns the katz centrality for each node in a
    dictonary which is sorted on the keys(node) for a given graph
    
    Input: G: networkx.classes.graph.Graph
           i: int
                max_iter value for nx.katz_centrality()
    Output: dict
            key = node, value = katz centrality of the node
            (rounded to 3 decimal places)
    """

    adj = find_adj_matrix(G)

    # Determine the eigenvalues of the matrix
    eigen_values, eigen_vectors = np.linalg.eig(adj.todense())
    
    # find the value of alpha (0 < α < 1/k1)
    max_eigen_value = np.round(max(eigen_values),2).real
    alf = (1/max_eigen_value) - 0.05
    
    k_cent = nx.katz_centrality(G, alpha = alf, max_iter=i)
    
    katz_cent_dict = {}
    for node, k_cent in sorted(k_cent.items()):
        katz_cent_dict[node] = round(k_cent,3)

    return katz_cent_dict


def find_page_rank(G, i):
    
    """
    This function returns the page rank for each node in a dictonary
    which is sorted on the keys(node) for a given graph
    
    Input: G: networkx.classes.graph.Graph
           i: int
                max_iter value for nx.pagerank()
    Output: dict
            key = node, value = page rank of the node
            (rounded to 3 decimal places)
    """
    
    adj = find_adj_matrix(G) 
    
    # Construct the diagonal matrix D_ii
    d = np.diag([max(G.out_degree(node),1) for node in nx.nodes(G)])

    # Find largest eigenvalue of AD^{-1}.
    eigen_values, eigen_vectors = np.linalg.eig(np.dot(adj.todense(),np.linalg.inv(d)))
    
    max_eigen_value = np.round(max(eigen_values),3).real
    alf = (1/max_eigen_value) - 0.15
  
    page_rank = nx.pagerank(G, alpha=alf, max_iter=i)
    
    pg_rank_dict = {}
    for vertex, pg_rank in sorted(page_rank.items()):
        pg_rank_dict[vertex] = round(pg_rank,3)
        
    return pg_rank_dict


def find_betweenness_centrality(G):
    
    """
    This function returns the betweenness centrality for each node in 
    a dictonary which is sorted on the keys(node) for a given graph
    
    Input: G: networkx.classes.graph.Graph
    Output: dict
            key = node, value = betweenness centrality of the node
            (rounded to 3 decimal places)
    
    """
    
    bw_cent = nx.betweenness_centrality(G, k=nx.number_of_nodes(G))
    
    bw_cent_dict = {}
    for node, bw in sorted(bw_cent.items()):
        bw_cent_dict[node] = round(bw,3)
    return bw_cent_dict


def find_closeness_centrality(G):
    
    """
    This function returns the closeness centrality for each node in a
    dictonary which is sorted on the keys(node) for a given graph
    
    Input: networkx.classes.graph.Graph
    Output: dict
            key = node, value = closeness centrality of the node
            (rounded to 3 decimal places)
    
    """
        
    closeness_cent = nx.closeness_centrality(G)
    
    close_cent_dict = {}
    for node, c_cent in sorted(closeness_cent.items()):
        close_cent_dict[node] = round(c_cent,3)
            
    return close_cent_dict


def find_geodesic_distance(G):
    
    """
    This function returns all the geodesic distance between pair of
    nodes of the graph
    
    Input: networkx.classes.graph.Graph
    Output: list
    
    """
    
    geo_dis = [n for n in nx.shortest_path_length(G)]
    geo_dis_list = []

    for src in geo_dis:
           geo_dis_list += (g_dis for vertex, g_dis in src[1].items())
    
    return geo_dis_list


def find_diameter(G):
    
    """
    This function returns the diameter for the giant component
    of a given graph
    
    Input: networkx.classes.graph.Graph
    Output: int
    
    """
    # find the giant component of the random graph
    Gcc = max(nx.connected_components(G), key=len)
    Gl = G.subgraph(Gcc).copy()
        
    # calulate diameter 
    diameter = nx.diameter(Gl)
    
    return diameter


def find_llc(G):
    
    """
    This function returns the local clustering coefficient for each node
    of a given graph
    
    Input: networkx.classes.graph.Graph
    Output: dict
            key = node, value = local clustering coefficient of the node
            (rounded to 3 decimal places)

    
    """
    
    # Find the local clustering coefficient for each node:
    local_clus_coef = nx.clustering(G)
    
    llc_dict = {}
    for node, llc in sorted(local_clus_coef.items()):
        llc_dict[node] = round(llc,3)
    
    return llc_dict


def plot_distribution(in_list, pdf, cum, bw, title, x_label, y_label, x_scale, y_scale, c):
    
    """
    This function plots the histogram for given input list such as: list of degrees,
    list of eigenvalue centrality etc of the graph.
    
    Input:  in_list: list,
                dataset for which histogram is to be plotted
            pdf: bool,
                set as TRUE if density distribution is desired, otherwise FALSE
            cum: bool, 
                set as TRUE if cummulative distribution is desired, otherwise FALSE
            bw=float/int
                bin-width, to set bin size
            title: str,
                title for the plot
            x_label: str,
                x-axis label for the plot 
            y_label: str,
                y-axis lable for the plot
            x_scale: str,
                set type of x-axis scale such as: linear, log, symlog, logit
            y_scale: str,
                set type of x-axis scale such as: linear, log, symlog, logit
            c: str
                set the color of the bars
    
    Ouput: None
    
    """
    
    # Set bin size for plot
    bin_width = bw
    b = math.ceil((max(in_list) - min(in_list))/bin_width)
    
    # Configure figure size
    plt.figure(figsize=(20, 8))


    # Plot histogram
    plt.hist(in_list, density=pdf, cumulative=cum, color=c, bins=b)
    
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.title(f"{title}", size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.xlabel(f"{x_label}", size=15)
    plt.ylabel(f"{y_label}", size=15)
    plt.show()
    
    return None


def find_node_with_largest_val(centrality):
    
    """
    This function return the list of nodes with highest cenrality: degree,
    eigenvector, katz, pagerank, closeness and betweenness for a graph
    
    Input: 
        centrality: dict
            key = node, value = centrality of the node
    
    Output: list
    
    """                        
    tmp_dict = {}                       
    for n, c in centrality.items():
        try:
            tmp_dict[c].append(n)
        except:
            tmp_dict[c] = [n]
    
    max_val = max(tmp_dict.keys())
    nodes_with_max_val = tmp_dict[max_val]
                
    return nodes_with_max_val


def find_num_of_edges(n,k):
    
    '''
    It return the number of edges for given number of nodes and
    value of <k> for 20 iteration of a random graph.
    
    Input:  n: int
            number of nodes/vertices
    
    Output: list
    '''
    
    edges = []
    for _ in range(20):
        prob = k/(n-1)
        rG = nx.gnp_random_graph(n=n, p=prob)
        edges.append(nx.number_of_edges(rG))
    return edges

def plot_power_low_distribution(degree_list):
    
    """
    This function plot power low distribution of degrees for a given degree list of a graph.
    
    Input: list
    Output: None
    
    """
    result = powerlaw.Fit(degree_list, discrete=True, verbose=False)
    print(f"Value of alpha for the powerlaw distribution: {round(result.power_law.alpha,3)}","\n\n")

    # Configure figure size    
    plt.figure(figsize=(20,8))

    # Plot the degree distribution for real network
    plt.plot(*np.unique(degree_list, return_counts=True), 'ob')
    plt.title('Degree distribution of the graph', size=25)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree (log scale)', size=20)
    plt.ylabel('Count (log scale)', size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()
    
    print("\n\n")
    return None