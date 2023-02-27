import pandas as pd
from sklearn.cluster import KMeans

center_path = "cluster_centers.csv"
center_data = pd.read_csv(center_path, index_col = None).values

kmeans = KMeans(n_clusters= 8 , init = center_data)
