import pandas as pd 
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.data import loadlocal_mnist
from networkx.algorithms import centrality
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report, f1_score

from sklearn.model_selection import train_test_split


def print_graph(knn_graph_obj, img_name='none'):
        print('[INFO] Drawing graph')
        plt.figure(figsize=(18,18))
        nx.draw(knn_graph_obj, with_labels=True, font_weight='bold')
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



dataset = 'digits'
X_load, y_load = open_csv_dataframe(dataset)
#neighbors = int(np.sqrt(len(X_loadinho)))
neighbors = 5

X_load_train, X_load_test, y_load_train, y_load_test = train_test_split(X_load, y_load, train_size=0.7, random_state=42)


print('[INFO] Contructing graph')
knn_graph = kneighbors_graph(X_load_train, neighbors, mode = 'distance') #metrica de distância default = euclidiana

print('[INFO] Converting to network x graph object')
knn_graph_obj = nx.from_scipy_sparse_matrix(knn_graph, create_using=nx.Graph())

for perc_labeled in [0.01, 0.05, 0.1]:
        #print_graph(knn_graph_obj, dataset)
        #write_on_file(knn_graph_obj, dataset)

        degree_centrality_knn = pd.DataFrame.from_dict(centrality.degree_centrality(knn_graph_obj), orient='index', columns=['value'])

        node_toget_labels = degree_centrality_knn.sort_values(by = 'value', ascending = False).index[0:int(perc_labeled*len(degree_centrality_knn.index))]
        #node_toget_labels = [10, 20, 50, 55, 80, 99, 300, 413, 555, 762, 875, 1003, 1234, 1254, 1442]

        y_loadinho = y_load_train.filter(items = node_toget_labels, axis = 0)
        X_loadinho = X_load_train.filter(items = node_toget_labels, axis = 0)

        datasetinho = pd.concat([X_loadinho, y_loadinho], axis = 1)

        #datasetinho.to_csv('digits1.csv')

        X_loadao = X_load_train[~X_load_train.index.isin(node_toget_labels)]
        X_loadao['y'] = -1

        final_dataset = pd.concat([X_loadao, datasetinho], axis=0).sort_index()

        #final_dataset.to_csv('digits05_final.csv')

        lp_model = LabelSpreading()
        X = final_dataset.drop('y', axis = 1)
        labels = final_dataset['y']
        lp_model.fit(X, labels)

        #predicted_labels = lp_model.transduction_[X_load_train.index]

        #print(classification_report(y_load[y_load.index.isin(X_loadao.index)], predicted_labels))

        y_predict = lp_model.predict(X_load_test)
        print(f1_score(y_load_test, y_predict))
        txt_file = open(str(int(perc_labeled*100))+"perc_digits.txt","a") 
        txt_file.write(classification_report(y_load_test, y_predict))
        txt_file.close() 