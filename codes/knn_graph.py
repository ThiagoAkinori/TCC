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
from label_propagation import LGC

def print_graph(knn_graph_obj, node_labels, img_name='none' ):
        print('[INFO] Drawing graph')
        plt.figure(figsize=(18,18))
        pos = nx.spring_layout(knn_graph_obj)       
        nx.draw(knn_graph_obj,pos, with_labels=True, font_weight='bold')
        
        nx.draw_networkx_nodes(knn_graph_obj,pos, nodelist=node_labels, node_color="r")
        plt.savefig('../results/'+img_name+'.png', dpi=600)

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


def get_centrality_labels(knn_graph_obj, type='degree'):
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
        else:
                indexes = list(knn_graph_obj.nodes)
                #print(indexes)
                node_toget_labels = random.sample(indexes, int(perc_labeled*len(indexes)))
                #print(node_toget_labels)

        return node_toget_labels

dataset = 'digits'
X_load, y_load = open_csv_dataframe(dataset)
#neighbors = int(np.sqrt(len(X_loadinho)))
neighbors = 5

print('[INFO] Contructing graph')
knn_graph = kneighbors_graph(X_load, neighbors, mode = 'distance') #metrica de distância default = euclidiana

print('[INFO] Converting to network x graph object')
knn_graph_obj = nx.from_scipy_sparse_matrix(knn_graph, create_using=nx.Graph())

metric_list = ['degreee','closeness','betweenness','random']

scores = pd.DataFrame(columns=['centrality', 'label_percentage', 'score'])

for metric in metric_list:
        for perc_labeled in [0.01, 0.05, 0.1]:
                #write_on_file(knn_graph_obj, dataset)
                node_toget_labels = get_centrality_labels(knn_graph_obj, metric)
        
                print_graph(knn_graph_obj, node_toget_labels, str(int(perc_labeled*100))+'_'+metric+'_'+dataset)

                y_loadinho = y_load.filter(items = node_toget_labels, axis = 0).sort_index()
                X_loadinho = X_load.filter(items = node_toget_labels, axis = 0).sort_index()

                lgc = LGC(nx.to_scipy_sparse_matrix(knn_graph_obj,nodelist=X_load.index))

                lgc.fit(X_loadinho.index, y_loadinho)

                #datasetinho = pd.concat([X_loadinho, y_loadinho], axis = 1)

                #datasetinho.to_csv('digits1.csv')

                X_loadao = X_load[~X_load.index.isin(node_toget_labels)]
                y_loadao = y_load[~y_load.index.isin(node_toget_labels)]
                #X_loadao['y'] = -1

                y_predict = lgc.predict(X_loadao.index)
                #final_dataset = pd.concat([X_loadao, datasetinho], axis=0).sort_index()

                #final_dataset.to_csv('digits'+str(int(perc_labeled*100))+'_'+metric+'_final.csv')

                #lp_model = LabelSpreading()
                #X = final_dataset.drop('y', axis = 1)
                #labels = final_dataset['y']
                #lp_model.fit(X, labels)

                #predicted_labels = lp_model.transduction_[X_load_train.index]

                #print(classification_report(y_load[y_load.index.isin(X_loadao.index)], predicted_labels))

                #y_predict = lp_model.predict(X_load_test)
                score = f1_score(y_loadao, y_predict)
                print(score)
                scores = scores.append(pd.DataFrame(
                        {'centrality':[metric], 
                        'label_percentage':[perc_labeled], 
                        'score':[score]}))
                #txt_file = open(str(int(perc_labeled*100))+"perc_digits_"+metric+".txt","a") 
                #txt_file.write(classification_report(y_load_test, y_predict))
                #txt_file.close() 

scores.to_csv(dataset+'_scores.csv', index=False)  