from cluster import (
  make_clusters, 
  plot_clusters, 
  plot_multipanel)

"""
Cluster Generation
"""

# here we are making tightly clustered data by lowering the scale param
t_clusters, t_labels = make_clusters(scale=0.3) 

# here we are making loosely clustered data by increasing the scale param
l_clusters, l_labels = make_clusters(scale=2) 

# here we are making many clusters by adjusting the `k` param
m_clusters, m_labels = make_clusters(k=10)

# here we are directly controlling the dimensionality of our data 
#   1000 observations 
#   200 features 
#   3 clusters)
d_clusters, d_labels = make_clusters(n=1000, m=200, k=3)


"""
Cluster Visualization
"""
# show the visualization for some clusters and their associated labels
#plot_clusters(t_clusters, t_labels)
plot_clusters(l_clusters, l_labels)

# show a multipanel visualization of clusters with true labels, predicted labels, and silhouette scores
# you will need to calculate these predicted labels with your kmeans implementation and the scores 
# with your silhouette score implementation
# plot_multipanel(t_clusters, t_labels, pred_labels, scores)