import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess():
    data = pd.read_csv('mall_customers.csv').drop(['CustomerID', 'Gender'], axis=1).to_numpy()

    # scaling data
    mins = np.min(data, axis=0)
    maxes = np.max(data, axis=0)
    ranges = maxes - mins
    scaled_data = (data - mins) / ranges

    unscale_centroids = lambda centroids: centroids * ranges + mins

    return scaled_data, unscale_centroids

# returns matrix where each row is a centroid, making the matrix of size (k, # features) 
def fit(data: np.ndarray, k_means: int, epochs: int=None, min_epsilon: float=None):
    if not epochs and not min_epsilon:
        raise ValueError('Must provide atleast one stopping point')

    # conditions to know when to stop fitting or display epsilon
    done_epochs = lambda e: not epochs or e >= epochs
    done_epsilon = lambda eps: not min_epsilon or eps < min_epsilon
    stop = lambda e, eps: done_epochs(e) or done_epsilon(eps)


    # initialize centroids
    centroids = np.empty((k_means, data.shape[1]))

    # initialize one centroid randomly from the dataset
    init_idx = np.random.randint(data.shape[0])
    centroids[0] = data[init_idx]

    selected_inds = [init_idx]

    for k in range(1, k_means):
        
        distances = [np.min(np.sqrt(np.sum((centroids[:k] - vector) ** 2, axis=1))) for vector in np.delete(data, selected_inds, axis=0)]

        # data_len = data.shape[0] - len(selected_inds)

        probabilities = [distances[idx] / sum(distances) for idx in range(len(distances))]

        idx = np.random.choice(len(distances), p=probabilities)
        selected_inds.append(idx)
        centroids[k] = data[idx]
    

    epoch = 0
    
    while True:
        epoch += 1

        clusters = [[] for _ in range(k_means)]
        # cluster points
        idx = predict(centroids, data)
        for i, j in enumerate(idx):
            clusters[j].append(data[i])

        # calculate new centroids based on the means of the vectors in the current centroids
        # replaces centroids with no vectors with random point
        centroid_means = [np.array(cluster).mean(axis=0) if len(cluster) > 0 else data[np.random.randint(data.shape[0])] for cluster in clusters]

        new_centroids = np.array(centroid_means)

        # max distance between the old and new centroids
        epsilon = np.max(np.sqrt(np.sum((new_centroids - centroids) ** 2)))

        # if the number of epochs is reached or the distance between the centriods is less than the minimum, then stop fitting
        if stop(epoch, epsilon):
            return new_centroids
        centroids = new_centroids
        
def predict(centroids, vectors):
    return np.array([np.argmin(np.sqrt(np.sum((centroids - vector) ** 2, axis=1))) for vector in vectors])


def elbow_method(kmin, kmax, data, epochs):
    k_values = list(range(kmin, kmax + 1))
    inertia_values = []
    for k in k_values:
        centroids = fit(data, k, epochs=epochs)
        vector_centroid_idx = predict(centroids, data)
        # inertia is the sum of the squares of the euclidean distance between each vector and its centroid
        inertia_values.append(np.sum([(centroids[cidx] - data[vidx]) ** 2 for vidx, cidx in enumerate(vector_centroid_idx)]))
    
    plt.plot(k_values, inertia_values)
    plt.xlabel('# Clusters')
    plt.ylabel('Inertia')
    plt.show()

data, unscale_centroids = preprocess()
centroids = fit(data, 6, epochs=1000)
np.save('centroids', unscale_centroids(centroids))

# -----------------------------------------------------------

# vectors = pd.read_csv('mall_customers.csv').drop(['CustomerID', 'Gender'], axis=1).to_numpy()

# centroids = np.load('centroids.npy')
# predictions = predict(centroids, vectors)

# color_map = ['r', 'g', 'b', 'y', 'm', 'c']

# colors = [color_map[p] for p in predictions]


# fig = plt.figure()

# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], c=colors)

# ax.set_xlabel('Age')
# ax.set_ylabel('Income')
# ax.set_zlabel('Spending Score')

# plt.show()