import numpy as np
import turtle
import pandas as pd

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def find_leaf_nodes(root):
    leaves = []

    def dfs(node):
        if not node:
            return
        if not node.left and not node.right:
            leaves.append(node.val)

        dfs(node.left)
        dfs(node.right)

    dfs(root)
    return leaves


def draw_tree(node, x, y, dx):
    if node is not None:
        # Draw the current node
        turtle.penup()
        turtle.goto(x, y)
        turtle.pendown()
        turtle.write(node.val, align='center', font=('Comic sans', 12, 'normal'))

        #Left subtree
        if node.left is not None:
            turtle.goto(x, y)
            turtle.pendown()
            turtle.goto(x - dx, y - 60)
            draw_tree(node.left, x - dx, y - 60, dx / 2)
        
        #Right subtree
        if node.right is not None:
            turtle.penup()
            turtle.goto(x, y)
            turtle.pendown()
            turtle.goto(x + dx, y - 60)
            draw_tree(node.right, x + dx, y - 60, dx / 2)


def k_means(data, k, tolerance=1e-3, MAX_ITER=500):
    centroids_idx = np.random.choice(data.shape[0], size=k, replace=False)
    centroids = data[centroids_idx]
    labels = _get_labels(data, centroids)

    err = float('inf')
    iter = 0
    while err > tolerance and iter < MAX_ITER:
        new_centroids = _get_centroids(data, labels, k)
        err = np.mean(np.sum(np.abs(new_centroids - centroids), axis=1))
        centroids = new_centroids
        labels = _get_labels(data, centroids)

    return labels


def _get_centroids(data, labels, k):
    centroids = np.array([np.mean(data[labels == i], axis=0) for i in range(k)])
    return centroids


def _get_labels(data, centroids):
    # d_mesh, c_mesh = np.meshgrid(data, centroids)
    dist = np.linalg.norm(data[:, np.newaxis] - centroids[np.newaxis, :], axis=2)
    labels = np.argmin(dist, axis=1)
    return labels


def complete_linkage(data):
    clusters = [TreeNode(i) for i in range(data.shape[0])]
    dist_matrix = _dist_matrix(data)
    n = len(clusters)

    while n > 1:
        max_dist = float('-inf')
        pair = (0, 0)

        for i in range(n):
            for j in range(i + 1, n):
                cluster_dist = _complete_linkage_dist(dist_matrix, clusters[i], clusters[j])

                if cluster_dist > max_dist:
                    max_dist = cluster_dist
                    pair = (i, j)

        cluster_i, cluster_j = pair
        new_cluster = TreeNode(round(max_dist, 1), clusters[cluster_i], clusters[cluster_j])
        clusters.append(new_cluster)

        del clusters[max(cluster_i, cluster_j)]
        del clusters[min(cluster_i, cluster_j)]

        n = len(clusters)

    return clusters[0]


def _complete_linkage_dist(dist_matrix, cluster_i, cluster_j):
    points_i = find_leaf_nodes(cluster_i)
    points_j = find_leaf_nodes(cluster_j)
    return np.max(np.array([dist_matrix[points_i, point_j] for points_i in points_i for point_j in points_j]))


def single_linkage(data):
    clusters = [TreeNode(i) for i in range(data.shape[0])]
    dist_matrix = _dist_matrix(data)
    n = len(clusters)

    while n > 1:
        min_dist = float('inf')
        pair = (0, 0)

        for i in range(n):
            for j in range(i + 1, n):
                cluster_dist = _single_linkage_dist(dist_matrix, clusters[i], clusters[j])

                if cluster_dist < min_dist:
                    min_dist = cluster_dist
                    pair = (i, j)

        cluster_i, cluster_j = pair
        new_cluster = TreeNode(round(min_dist, 1), clusters[cluster_i], clusters[cluster_j])
        clusters.append(new_cluster)

        del clusters[max(cluster_i, cluster_j)]
        del clusters[min(cluster_i, cluster_j)]

        n = len(clusters)

    return clusters[0]


def _single_linkage_dist(dist_matrix, cluster_i, cluster_j):
    points_i = find_leaf_nodes(cluster_i)
    points_j = find_leaf_nodes(cluster_j)
    return np.min(np.array([dist_matrix[points_i, point_j] for points_i in points_i for point_j in points_j]))


def _dist_matrix(data):
    n = data.shape[0]
    matr = np.zeros((n,n))
    
    for i in range(n):
        for j in range(i+1, n):
            matr[i, j] = _dist(data[i], data[j])
            matr[j, i] = matr[i, j]
    return matr


def _dist(x, y, axis=None):
    return np.linalg.norm(x - y, axis)

data = path = "./hw1/dataset1.csv"

df = pd.read_csv(path)

data = df.to_numpy()[:,1:]
k_means(data, k=6)


# root = complete_linkage(data[:10])

# turtle.speed(0) 
# turtle.hideturtle()
# draw_tree(root, 0, 300, 80)
# turtle.done()