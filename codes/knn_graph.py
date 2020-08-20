import pandas as pd 
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.data import loadlocal_mnist
from networkx.algorithms import centrality, clustering
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report, f1_score

from sklearn.model_selection import train_test_split
from label_propagation import LGC,HMN

def print_graph(knn_graph_obj, node_labels, img_name='none' ):
        print('[INFO] Drawing graph')
        plt.figure(figsize=(18,18))
        pos = nx.spring_layout(knn_graph_obj)       
        nx.draw(knn_graph_obj,pos, with_labels=True, font_weight='bold')
        
        nx.draw_networkx_nodes(knn_graph_obj,pos, nodelist=node_labels, node_color="r")
        plt.savefig('../results/'+img_name+'.png', dpi=600)
        plt.close()

def write_on_file(knn_graph_obj, obj_name='none'):
        print('[INFO] Writing file')
        nx.write_adjlist(knn_graph_obj,'../results/'+obj_name+'.adjlist')


def open_mnist():
        X_load, y = loadlocal_mnist(
                images_path='/home/thiagosato/Documentos/TCC/datasets/mnist/train-images.idx3-ubyte', 
                labels_path='/home/thiagosato/Documentos/TCC/datasets/mnist/train-labels.idx1-ubyte'
        )

        X_loadinho = X_load[0:10]
        return X_loadinho

def open_csv_dataframe(dataset):
        path = '../datasets/'+dataset+'/'+dataset+'.csv'

        X_loadinho = pd.read_csv(path)
        y_load = X_loadinho['y']
        X_load = X_loadinho.drop('y', axis = 1)
        
        if dataset == 'digits':
                y_load.loc[y_load==-1] = 0
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

for algorithm in ['hmn', 'lgc']:
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

                print('[INFO] Converting to network x graph object')
                knn_graph_obj = nx.from_scipy_sparse_matrix(knn_graph, create_using=nx.Graph())
                
                #
                metric_list = ['degree','closeness','betweenness', 'clustering','random']

                scores = pd.DataFrame(columns=['centrality']+[str(x)+'%' for x in range(1,21,1)])

                count_labels = pd.DataFrame(columns=['centrality','percentage']+y_load.unique().tolist())

                perc_labels = [x/100 for x in range(1,21,1)]
                #algorithm = 'hmn'
                for metric in metric_list:
                        print('[INFO] Metric '+ metric)
                        score_list = []
                        count_list = pd.DataFrame(columns=['percentage']+y_load.unique().tolist())
                        for perc_labeled in perc_labels:
                                print('[INFO] Training with '+ str(100*perc_labeled)+" label percentage")
                                subscore_list = []
                                subcount_list = pd.DataFrame(columns=y_load.unique())
                        
                                if metric == 'random':
                                        n = 5
                                else:
                                        n = 1
                                for i in range(n):
                                        #write_on_file(knn_graph_obj, dataset)
                                        node_toget_labels = get_centrality_labels(knn_graph_obj, perc_labeled, metric)
                                
                                        print_graph(knn_graph_obj, node_toget_labels, dataset+'/'+metric+'/'+str(int(perc_labeled*100))+'_'+metric+'_'+dataset)

                                        y_loadinho = y_load.filter(items = node_toget_labels, axis = 0).sort_index()
                                        X_loadinho = X_load.filter(items = node_toget_labels, axis = 0).sort_index()
                                        
                                        #print(y_loadinho.value_counts().to_frame().T)
                                        subcount_list=subcount_list.append(y_loadinho.value_counts().to_frame().T, ignore_index=True)
                                        #print(subcount_list)

                                        if algorithm == 'lgc':
                                                model = LGC(nx.to_scipy_sparse_matrix(knn_graph_obj,nodelist=X_load.index))
                                        else:
                                                model = HMN(nx.to_scipy_sparse_matrix(knn_graph_obj,nodelist=X_load.index))

                                        model.fit(X_loadinho.index, y_loadinho)

                                        #datasetinho = pd.concat([X_loadinho, y_loadinho], axis = 1)

                                        #datasetinho.to_csv('digits1.csv')

                                        X_loadao = X_load[~X_load.index.isin(node_toget_labels)]
                                        y_loadao = y_load[~y_load.index.isin(node_toget_labels)]
                                        #X_loadao['y'] = -1

                                        y_predict = model.predict(X_loadao.index)
                                        score = f1_score(y_loadao, y_predict)
                                        subscore_list.append(score)
                                
                                perc_df = pd.DataFrame({'percentage':[perc_labeled]})
                                count_list = count_list.append(pd.concat([perc_df,np.mean(subcount_list).to_frame().T], axis=1), ignore_index=True)
                                score_list.append(np.mean(np.array(subscore_list)))
                                print(np.mean(np.array(subscore_list)))
                        count_list['centrality'] = metric
                        count_labels = count_labels.append(count_list, ignore_index=True)
                        scores = scores.append(pd.DataFrame([[metric]+score_list], columns=['centrality']+[str(x)+'%' for x in range(1,21,1)]))
                name = algorithm+'_'+dataset+'_scores.csv'
                print('[INFO] Saving in '+name)
                count_labels.to_csv('count_labels_'+name, index=False)
                scores.to_csv(name, index=False)  