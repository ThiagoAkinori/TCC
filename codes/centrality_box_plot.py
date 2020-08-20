import pandas as pd 
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.data import loadlocal_mnist
from networkx.algorithms import centrality, clustering
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report, f1_score
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

def open_csv_dataframe(dataset):
    path = '../datasets/'+dataset+'/'+dataset+'.csv'

    X_loadinho = pd.read_csv(path)
    y_load = X_loadinho['y']
    X_load = X_loadinho.drop('y', axis = 1)
        
    return X_load, y_load


def get_centrality_labels(knn_graph_obj, type='degree'):
    import random

    if type == 'degree':
        node_toget_labels = pd.DataFrame.from_dict(centrality.degree_centrality(knn_graph_obj), orient='index', columns=['value'])
    elif type == 'closeness':
        node_toget_labels = pd.DataFrame.from_dict(centrality.closeness_centrality(knn_graph_obj), orient='index', columns=['value'])
    elif type == 'betweenness':
        node_toget_labels = pd.DataFrame.from_dict(centrality.betweenness_centrality(knn_graph_obj), orient='index', columns=['value'])
    elif type == 'clustering':
        node_toget_labels = pd.DataFrame.from_dict(clustering(knn_graph_obj), orient='index', columns=['value'])
    else:
        node_toget_labels = list(knn_graph_obj.nodes)
            #print(node_toget_labels)

    return node_toget_labels

def disconnect_different_labels(knn_graph_obj, nodes, labels):
        
    for reference_node in nodes:
        for node in nodes:
            if knn_graph_obj.has_edge(reference_node, node):
                reference_node_label = labels[labels.index == reference_node].values[0]
                node_label = labels[labels.index == node].values[0]
                if reference_node_label != node_label:
                    knn_graph_obj.remove_edge(reference_node, node)
    return knn_graph_obj




for dataset in ['USPS','COIL','g241c','g241n','digits']:

    print('[INFO] Dataset '+dataset)
    
    X_load, y_load = open_csv_dataframe(dataset)
    #neighbors = int(np.sqrt(len(X_loadinho)))
    neighbors = 5
    y_load.loc[y_load == -1] = 0
    #print(y_load)
    print('[INFO] Contructing graph')
    knn_graph = kneighbors_graph(X_load, neighbors, mode = 'distance') #metrica de distância default = euclidiana

    print('[INFO] Converting to network x graph object')
    knn_graph_obj = nx.from_scipy_sparse_matrix(knn_graph, create_using=nx.Graph())
    knn_graph_obj = max(nx.connected_component_subgraphs(knn_graph_obj), key=len)
    
    metric_list = ['degree','closeness','betweenness', 'clustering']
    df = pd.DataFrame(columns=['metric', 'value'])

    for metric in metric_list:
        print('[INFO] Calculating Metric '+metric)
        centrality_list = get_centrality_labels(knn_graph_obj, metric)
        print('[INFO] Adjusting Scaling')
        adj_df = (centrality_list['value']-centrality_list['value'].min())/(centrality_list['value'].max()-centrality_list['value'].min())
        
        df_metric = pd.DataFrame()

        df_metric['value'] = adj_df
        df_metric['metric'] = metric

        df = df.append(df_metric)
    print('[INFO] Drawing graph')
    plt.figure(figsize=(12,12))
    sns.boxplot(x = 'value', y = 'metric', data = df)
    img_name = dataset
    plt.savefig('boxplot/'+img_name+'.png')
    plt.close()