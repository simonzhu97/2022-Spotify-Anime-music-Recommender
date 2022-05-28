"""
Use a clusering model to cluster the anime songs
"""
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score, pairwise_distances


def get_models_dict(df, cols, K_range, seed=42, return_list=False):
    """
    run several K-means models with different number of clusters K on a dataframe
    df: dataframe to work on
    cols: features to include in the K-Means algorithm
    K_range: range of number of clusters to try
    """
    
    model_dict = {}
    for k in K_range:
        mod = KMeans(n_clusters=k, random_state=seed).fit(df[cols])
        model_dict[str(k)] = mod
    res = model_dict
    
    if return_list:
        ls_mods = list(model_dict.values())
        res = (model_dict, ls_mods)
    return res

def plot_models_performance(df, cols, mod_list, K_range):
    """
    Plot silhoutte 
    """
    within_ss = [i.inertia_ for i in mod_list]
    silhouette_list = [silhouette_score(df[cols], i.labels_) for i in mod_list]
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    axs[0].plot(K_range, within_ss, color='red')
    axs[1].plot(K_range, silhouette_list, color='orange')
    
    for i in range(2): 
        axs[i].set_xlabel('number of clusters')
    for idx, name in zip([0,1,2], ['inertia', 'silhouette score']): 
        axs[idx].set_ylabel(name)
    
    plt.show()