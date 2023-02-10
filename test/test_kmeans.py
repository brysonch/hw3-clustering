# Write your k-means unit tests here
import pytest
from cluster import (KMeans, make_clusters)

def test_kmeans():

    try: 
        km_none = KMeans(k=0)
        none_test = False
    except:
        none_test = True

    try:
        km_noint = KMeans(k="bad")
        noint_test = False
    except TypeError:
        noint_test = True

    try: 
        k = 10
        clusters, labels = make_clusters(n=9, k=k, scale=1)
        km_lessobs = KMeans(k=k)
        km_lessobs.fit(clusters)
        lessobs_test = False
    except:
        lessobs_test = True

    assert none_test = True, "k must be greater than 0"
    assert noint_test = True, "k must be an integer"
    assert lessobs_test = True, "n observations must be greater than or equal to k clusters"
    
    
