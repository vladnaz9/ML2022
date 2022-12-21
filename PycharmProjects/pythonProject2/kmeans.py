import pandas as pd
import plotly.express as px
import random
import math
from matplotlib import pyplot as plt
import numpy as np


def random_point(n):
    points = []
    for i in range(n):
        points.append(np.random.randint(1, 100, 2))
    return points


def dist(p_i, p_j):
    return np.sqrt((p_i[0] - p_j[0]) ** 2 + (p_i[1] - p_j[1]) ** 2)


def dist2(p_i, p_j):
    return (p_i[0] - p_j[0]) ** 2 + (p_i[1] - p_j[1]) ** 2


def init_centroids(points, k):
    x_c = 0
    y_c = 0
    for i in range(len(points)):
        x_c += points[i][0]
        y_c += points[i][1]
    x_c /= len(points)
    y_c /= len(points)
    R = 0
    for i in range(len(points)):
        if R < dist([x_c, y_c], points[i]):
            R = dist([x_c, y_c], points[i])
    centroids = []
    for i in range(k):
        x_cntr = R * (np.cos(2 * np.pi * i / k)) + x_c
        y_cntr = R * (np.sin(2 * np.pi * i / k)) + y_c
        centroids.append([x_cntr, y_cntr])
    return centroids


def plot(points, centroids, clusters):
    clr = ['b', 'g', 'y', 'k']
    colors = []
    points_x = []
    points_y = []
    for i in range(len(points)):
        points_x.append(points[i][0])
        points_y.append(points[i][1])

    df = pd.DataFrame({'x': points_x,
                       'y': points_y,
                       'cluster': clusters})
    fig = px.scatter(df, x='x', y='y', color='cluster')
    fig.show()


def find_nearest(points, centroids):
    cluster = np.zeros(len(points))
    for i in range(len(points)):
        min_dist = np.infty
        for j in range(len(centroids)):
            if min_dist > dist(points[i], centroids[j]):
                min_dist = dist(points[i], centroids[j])
                cluster[i] = j
    return cluster


def get_centroids(points, cluster):
    centroids_dist = dict(zip(cluster, [None] * len(cluster)))
    for i in range(len(points)):
        dist_sum = centroids_dist.get(cluster[i])
        if (dist_sum is not None):
            dist_sum[0] += points[i][0]  # x
            dist_sum[1] += points[i][1]  # y
            dist_sum[2] = dist_sum[2] + 1  # count
        else:
            dist_sum = []
            dist_sum.append(points[i][0])  # x
            dist_sum.append(points[i][1])  # y
            dist_sum.append(1)  # count
        centroids_dist.update({cluster[i]: dist_sum})

    centroids = []
    for item in centroids_dist.items():
        centroids.append([item[1][0] / item[1][2], item[1][1] / item[1][2]])

    return centroids


def kmeans(points, k):
    centroids = init_centroids(points, k)
    cluster = find_nearest(points, centroids)


    while True:
        centroids = get_centroids(points, cluster)
        new_cluster = find_nearest(points, centroids)
        flag = True
        # plot(points, centroids, new_cluster)
        for i in range(len(cluster)):
            if (cluster[i] != new_cluster[i]):
                flag = False
                return [new_cluster, centroids]

        cluster = new_cluster
        if (flag):
            break
        iterations = iterations + 1
    print(f"Model is successfily built with iterations number = {iterations}. Result: ")


if __name__ == '__main__':
    iterations = 1
    n = 100
    points = random_point(n)
    K = len(points)
    dist_hist = []
    for k in range(2, K):
        result = kmeans(points, k)
        clusters = result[0]
        centroids = result[1]
        sqrt_clusters_dist = dict(zip(clusters, [None] * len(clusters)))
        dist_sum = 0
        for i in range(len(points)):
            cluster = clusters[i]
            сoordinates = points[i]
            cen_coordinates = centroids[int(cluster)]
            dist_sum = dist_sum + dist(сoordinates, cen_coordinates)
        dist_hist.append(dist_sum)
    plot(points, centroids, clusters)
    plt = px.line(x=[x for x in range(2, K)], y=dist_hist, title='Elbow curve')
    plt.show()

    min = abs(dist_hist[1] - dist_hist[2]) / abs(dist_hist[0] - dist_hist[1])
    k = 1
    for i in range(1, len(dist_hist) - 2):
        d = abs(dist_hist[i] - dist_hist[i + 1]) / abs(dist_hist[i - 1] - dist_hist[i])
        if (d < min):
            min = d
            k = i + 2
    print(f'Optmal K = {k}')




