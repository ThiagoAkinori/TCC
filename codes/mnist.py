import numpy as np
from mlxtend.data import loadlocal_mnist
from scipy.spatial.distance import euclidean
import threading
from threading import BoundedSemaphore
import pandas as pd

def concurrent_euclidian_distance_thread(array_a, lenght, i, X_load, euclidian_dist):
    pool_sema.acquire()
    print('[INFO] Initiating row '+str(i))
    for j in range(0, lenght):
        #print('[INFO] Calculating euclidian distance from '+str(i)+' and '+str(j))
        euclidian_dist[i][j] = euclidean(array_a, X_load[j])
    print('[INFO] Finalizing row '+str(i))
    pool_sema.release()



def concurrent_euclidian_distance(X_load):

    euclidian_dist = np.ndarray(shape=(len(X_load),len(X_load)))

    threads = list()
    for i in range(0, len(X_load)):
        lenght = len(X_load)
        x = threading.Thread(target=concurrent_euclidian_distance_thread, args=(X_load[i], lenght, i, X_load, euclidian_dist))
        threads.append(x)
        x.start()
    
    for x in threads:
        x.join()


    return euclidian_dist


def calculate_euclidian_distance(X_load):

    euclidian_dist = np.ndarray(shape=(len(X_load),len(X_load)))

    for i in range(0, len(X_load)):
        for j in range(0, len(X_load)):
            print('[INFO] Calculating euclidian distance from '+str(i)+' and '+str(j))
            if i != j:
                euclidian_dist[i][j] = euclidean(X_load[i], X_load[j])
            else:
                euclidian_dist[i][j] = 100
    return euclidian_dist


X_load, y = loadlocal_mnist(
        images_path='/home/thiagosato/Documentos/TCC/datasets/mnist/train-images.idx3-ubyte', 
        labels_path='/home/thiagosato/Documentos/TCC/datasets/mnist/train-labels.idx1-ubyte'
)

#X_loadinho = X_load[0:10]
X_loadinho = [[0,1,1], [0,1,1], [1,1,0], [0,0,1], [0,1,0], [0,0,0], [0,0,0], [1,1,1], [1,1,0], [1,0,0]]
maxconnections = 8
pool_sema = BoundedSemaphore(value=maxconnections)
#euclidian_dist = concurrent_euclidian_distance(X_loadinho)
euclidian_dist = calculate_euclidian_distance(X_loadinho)

#print(euclidian_dist.shape)

#for i in range(len(X_loadinho)):
#    euclidian_dist[i].mean()
f_name = "teste.csv"
print('[INFO] Writing distances on file '+ f_name)
pd.DataFrame(euclidian_dist).to_csv(f_name, index = False)


