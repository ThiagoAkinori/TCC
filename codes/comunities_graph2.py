import pandas as pd 
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import centrality, clustering
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report, f1_score
import community

def open_csv_dataframe(dataset):
    path = '../datasets/'+dataset+'/'+dataset+'.csv'

    X_loadinho = pd.read_csv(path)
    y_load = X_loadinho['y']
    X_load = X_loadinho.drop('y', axis = 1)
    return X_load, y_load


def get_centrality_labels(knn_graph_obj, perc_labeled, type='degree'):
        import random

        if type == 'degree':
                degree_centrality_knn = pd.DataFrame.from_dict(centrality.degree_centrality(knn_graph_obj), orient='index', columns=['value'])

                node_toget_labels = degree_centrality_knn.sort_values(by = 'value', ascending = False).index[0:int(perc_labeled*len(degree_centrality_knn.index))].tolist()
        elif type == 'closeness':
                closeness_centrality_knn = pd.DataFrame.from_dict(centrality.closeness_centrality(knn_graph_obj), orient='index', columns=['value'])

                node_toget_labels = closeness_centrality_knn.sort_values(by = 'value', ascending = False).index[0:int(perc_labeled*len(closeness_centrality_knn.index))].tolist()
        elif type == 'betweenness':
                betweenness_centrality_knn = pd.DataFrame.from_dict(centrality.betweenness_centrality(knn_graph_obj), orient='index', columns=['value'])

                node_toget_labels = betweenness_centrality_knn.sort_values(by = 'value', ascending = False).index[0:int(perc_labeled*len(betweenness_centrality_knn.index))].tolist()
        elif type == 'katz':
                katz_centrality_knn = pd.DataFrame.from_dict(centrality.katz_centrality(knn_graph_obj), orient='index', columns=['value'])

                node_toget_labels = katz_centrality_knn.sort_values(by = 'value', ascending = False).index[0:int(perc_labeled*len(katz_centrality_knn.index))].tolist()
        elif type == 'clustering':
                clustering_knn = pd.DataFrame.from_dict(clustering(knn_graph_obj), orient='index', columns=['value'])

                node_toget_labels = clustering_knn.sort_values(by = 'value', ascending = False).index[0:int(perc_labeled*len(clustering_knn.index))].tolist()
        else:
                indexes = list(knn_graph_obj.nodes)
                #print(indexes)
                node_toget_labels = random.sample(indexes, int(perc_labeled*len(indexes)))
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
    #dataset = 'digits', 'USPS'
    X_load, y_load = open_csv_dataframe(dataset)
    #neighbors = int(np.sqrt(len(X_loadinho)))
    neighbors = 5
    y_load.loc[y_load == -1] = 0
    #print(y_load)
    print('[INFO] Contructing graph')
    knn_graph = kneighbors_graph(X_load, neighbors, mode = 'distance') #metrica de distância default = euclidiana

    metric_list = ['degree','closeness','betweenness', 'clustering','random']

    count_labels = pd.DataFrame(columns=['centrality', 'comunity','percentage'])

    perc_labels = [x/100 for x in range(1,21,1)]
    #algorithm = 'hmn'
    for metric in metric_list:
        print('[INFO] Metric '+ metric)
        count_list = pd.DataFrame(columns=['percentage','comunity', 'label','node'])
                
        for perc_labeled in perc_labels:
            
            print('[INFO] Converting to network x graph object')
            knn_graph_obj = nx.from_scipy_sparse_matrix(knn_graph, create_using=nx.Graph())
            knn_graph_obj = max(nx.connected_component_subgraphs(knn_graph_obj), key=len)
            print('[INFO] Training with '+ str(100*perc_labeled)+" label percentage")

            node_toget_labels = get_centrality_labels(knn_graph_obj, perc_labeled, metric)

            y_loadinho = y_load.filter(items = node_toget_labels, axis = 0).sort_index()
            X_loadinho = X_load.filter(items = node_toget_labels, axis = 0).sort_index()
            print('[INFO] Desconecting connected nodes with different classes')
            knn_graph_obj = disconnect_different_labels(knn_graph_obj, node_toget_labels, y_loadinho)
            
            print('[INFO] Finding communities and building comunities graph')
            partitions = community.best_partition(knn_graph_obj)
            comunities = pd.DataFrame.from_dict(partitions, orient='index').reset_index()
            
            comunities.columns = ['node', 'comunity']
            
            #df = comunities.groupby(['comunity']).agg('count').reset_index()

            #df.to_csv( 'comunities2/'+metric+'_'+dataset+'_'+str(int(100*perc_labeled))+'_comunities.csv', index=False)
            
            node_labels = y_loadinho.copy().reset_index()
            node_labels.columns = ['node', 'label']

            node_labels = node_labels.merge(comunities, left_on='node', right_on='node')

            count_node_labels = node_labels.groupby(['label','comunity']).agg('count').reset_index()
            count_node_labels['percentage'] = perc_labeled


            count_list = count_list.append(count_node_labels, ignore_index=True)
        count_list['centrality'] = metric
        count_labels = count_labels.append(count_list, ignore_index=True)
    name = dataset+'_scores.csv'
    count_labels.to_csv('comunities2/'+'comunity_'+name, index=False)