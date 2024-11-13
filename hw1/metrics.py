import numpy as np
import pandas as pd

def f_1(data : pd.DataFrame):
    centroids = data.groupby('label').mean()
    centroids = centroids.rename(columns={'0' : 'x', '1' : 'y'})
    new_df = data.copy(deep=True)
    new_df = new_df.rename(columns={'0' : 'x', '1' : 'y'})
    new_df = new_df.merge(centroids, on='label', suffixes=('', '_centroid'))
    new_df['squared_dist'] = (new_df['x'] - new_df['x_centroid'])**2 + (new_df['y'] - new_df['y_centroid'])**2
    total_sum = new_df.groupby('label')['squared_dist'].sum().sum()
    return total_sum

def f_2(data):
    pass

def f_3(data):
    pass


path = "./hw1/complete.csv"

df = pd.read_csv(path)

f_1(df)