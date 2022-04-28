import numpy as np
import matplotlib.pyplot as plt

def K_means_cluster(k, data):
    # 1. randomly select k data points as
    #    centroids
    centroids = []
    for i in range(k):
        mean = np.random.uniform(-3,3,size=2)
        centroids.append(mean)

    dif = 1
    tol= 0.05
    # 3. repeat until the centroids converge:
    while (i <= 10) and dif > tol: #TODO: add condition
        previous_centroids = np.array(centroids)

        # 2. calculate the distance between
        #    each data point and each centroid
        d = []
        for dp in data:
            distance = [np.linalg.norm(dp - ci) for ci in centroids]
            d.append(distance)
        d = np.array(d)


        # 4. assign each data point to the cluster with the nearest centroid
        clusters = [[], []]
        for i, dp in enumerate(d):
            closest_centroid = np.argmin(dp)
            clusters[closest_centroid].append(list(data[i]))

        fig = plt.figure()
        ax1 = fig.add_subplot()
        ax1.scatter([row[0] for row in clusters[0]], [row[1] for row in clusters[0]], c='b', label='first')
        ax1.scatter([row[0] for row in clusters[1]], [row[1] for row in clusters[1]], c='r', label='second')
        plt.legend(loc='upper left');
        plt.show()

        # 5. recalculate the centroid of each cluster
        for i, c in enumerate(clusters):
            centroids[i] = np.mean(c,axis=0)

        dif = np.linalg.norm(centroids-np.array(previous_centroids))
        print("dif", dif)

