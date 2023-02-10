# write your silhouette score unit tests here
import pytest
from cluster import (Silhouette, make_clusters)

def test_silhouette_small():

    k = 4
    clusters, labels = make_clusters(k=k, scale=1)
    km = KMeans(k=k)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred)
    assert scores == 345.738521649737, "Silhouette does not work on small k"

def test_silhouette_large():

    k = 20
    clusters, labels = make_clusters(n=10000, m=4, k=k, scale=1)
    km = KMeans(k=k)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred)
    assert scores == 5329.559959117421, "Silhouette does not work on large k"
