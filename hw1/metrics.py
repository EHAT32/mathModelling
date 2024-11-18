import numpy as np
import pandas as pd
from itertools import combinations

def f_1(data : pd.DataFrame):
    centroids = data.groupby('label').mean()
    centroids = centroids.rename(columns={0 : 'x', 1 : 'y'})
    new_df = data.copy(deep=True)
    new_df = new_df.rename(columns={0 : 'x', 1 : 'y'})
    new_df = new_df.merge(centroids, on='label', suffixes=('', '_centroid'))
    new_df['squared_dist'] = (new_df['x'] - new_df['x_centroid'])**2 + (new_df['y'] - new_df['y_centroid'])**2
    total_sum = new_df.groupby('label')['squared_dist'].sum().sum()
    return total_sum

def calculate_squared_distances(group):
    total_squared_distance = 0
    # Generate all unique pairs of points in the group
    for (_, point1), (_, point2) in combinations(enumerate(group.itertuples()), 2):
        total_squared_distance += _dist(point1, point2)
    return total_squared_distance

def f_2(data : pd.DataFrame):

    new_df = data.copy(deep=True)
    new_df = new_df.rename(columns={0 : 'x', 1 : 'y'})
    group = new_df.groupby('label')
    total_dist = group.apply(calculate_squared_distances)
    return total_dist.sum()

def f_3(data):
    new_df = data.copy(deep=True)
    new_df = new_df.rename(columns={0 : 'x', 1 : 'y'})
    group = new_df.groupby('label')
    centroids = group.mean()
    label_count = new_df['label'].value_counts().sort_index()
    label_count = label_count.to_frame().to_numpy()
    new_df = new_df.merge(centroids, on='label', suffixes=('', '_centroid'))
    new_df['squared_dist'] = (new_df['x'] - new_df['x_centroid'])**2 + (new_df['y'] - new_df['y_centroid'])**2
    total_var = new_df.groupby('label')['squared_dist'].sum().sort_index()
    total_var = total_var.to_frame().to_numpy()
    variance = np.divide(total_var, label_count)
    return np.sum(variance)


def _dist(p1, p2):
    return (p1.x - p2.x)**2 + (p1.y - p2.y)**2

path = "./hw1/single.csv"

df = pd.read_csv(path)
print(df)

print(f_1(df), f_2(df), f_3(df))