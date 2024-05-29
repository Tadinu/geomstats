#!/usr/bin/env python
# coding: utf-8

# # Align all and Compute for Graphs
# 
# $\textbf{Lead Author: Anna Calissano}$

# Dear learner, 
# 
# the aim of the current notebook is to introduce the align all and compute as a learning method for graphs. The align all and compute allows to estimate the Frechet Mean, the Generalized Geodesic Principal Components and the Regression. In this notebook you will learn how use all the learning methods.

# In[1]:


import random

import networkx as nx

import geomstats.backend as gs

from geomstats.geometry.stratified.graph_space import GraphSpace
from geomstats.learning.aac import AAC

gs.random.seed(2020)


# Let's start by creating simulated data using `networkx`.

# In[2]:


graphset_1 = gs.array(
    [
        nx.to_numpy_array(nx.erdos_renyi_graph(n=5, p=0.6, directed=True))
        for i in range(10)
    ]
)
graphset_2 = gs.array(
    [
        nx.to_numpy_array(nx.erdos_renyi_graph(n=5, p=0.6, directed=True))
        for i in range(100)
    ]
)
graphset_3 = gs.array(
    [
        nx.to_numpy_array(nx.erdos_renyi_graph(n=3, p=0.6, directed=True))
        for i in range(1000)
    ]
)

nx.draw(nx.from_numpy_array(graphset_1[0]))


# ### A primer in space, metric and aligners

# The first step is to create the total space and then add quotient structure to it.

# In[3]:


total_space = GraphSpace(n_nodes=5)
total_space.equip_with_group_action()  # permutations by default

graph_space = total_space.equip_with_quotient_structure()


# By default, the total space comes equipped with the Frobenius metric (`MatricesMetric`) and graph space with a quotient metric.

# With the FAQ alignment and the default Frobenius norm on the total space, we match two graphs and a set of graphs to a base graph:

# In[4]:


permutated_graph = total_space.aligner.align(graphset_1[1], graphset_1[0])

permuted_graphs = total_space.aligner.align(graphset_1[1:3], graphset_1[0])


# To compute the distance we can either call the distance function:

# In[5]:


graph_space.metric.dist(graphset_1[0], graphset_1[1])


# Or, if matching has been already done, we can use the total space distance, to avoid computing the matching twice:

# In[6]:


total_space.metric.dist(graphset_1[0], permutated_graph)


# We can also align points to geodesics:

# In[7]:


init_point, end_point = graph_space.random_point(2)

geodesic_func = graph_space.metric.geodesic(init_point, end_point)

aligned_init_point = total_space.aligner.align_point_to_geodesic(
    geodesic_func, init_point
)

total_space.metric.dist(init_point, aligned_init_point)


# This short introduction should be enough to set you up for experimenting with the learning algorithms on graphs.

# ### Frechet Mean
# Reference: Calissano, A., Feragen, A., & Vantini, S. (2020). Populations of unlabeled networks: Graph space geometry and geodesic principal components. MOX Report.
# 
# Given $\{[X_1], \dots, [X_k]\}, [x_i] \in X/T$, we estimate the Frechet Mean using AAC consisting on two steps:
# 1. Compute $\hat{X}$ as arithmetic mean of $\{X_1, \dots, X_k\}, X_i \in X$ 
# 2. Using graph to graph alignment to find $\{X_1, \dots, X_k\}, X_i \in X$ optimally aligned with $\hat{X}$

# Let's instantiate the graph space.

# In[8]:


total_space = GraphSpace(n_nodes=5)
total_space.equip_with_group_action()
total_space.equip_with_quotient_structure();


# And now create the estimator, and fit the data.

# In[9]:


aac_fm = AAC(space=total_space, estimate="frechet_mean", max_iter=20)

fm = aac_fm.fit(graphset_2)

fm.estimate_


# ### Principal Components
# Reference: Calissano, A., Feragen, A., & Vantini, S. (2020). Populations of unlabeled networks: Graph space geometry and geodesic principal components. MOX Report.
# 
# We estimate the Generalized Geodesics Principal Components Analysis (GGPCA) using AAC. Given $\{[X_1], \dots, [X_k]\}, (s_i,[X_i]) \in X/T $ we are searching for:
# $\gamma: \mathbb{R}\rightarrow X/T$ generalized geodesic principal component capturing the majority of the variability of the dataset. The AAC for ggpca works in two steps: 
# 
# 1. finding $\delta: \mathbb{R}\rightarrow X$ principal component in the set of adjecency matrices $\{X_1, \dots, X_k\}, X_i \in X$ 
# 2. finding $\{X_1, \dots, X_k\}, X_i \in X$ as optimally aligned with respect to $\gamma$. The estimation required a point to geodesic aligment defined in the metric.

# As before:

# In[10]:


total_space = GraphSpace(n_nodes=3)
total_space.equip_with_group_action()
total_space.equip_with_quotient_structure();


# For GGPCA, we also need the point to geodesic aligner.

# Again, create the estimator and fit the data.

# In[11]:


aac_ggpca = AAC(space=total_space, estimate="ggpca", n_components=2)

aac_ggpca.fit(graphset_3);


# ## Regression
# Reference: Calissano, A., Feragen, A., & Vantini, S. (2022). Graph-valued regression: Prediction of unlabelled networks in a non-Euclidean graph space. Journal of Multivariate Analysis, 190, 104950.
# 
# We estimate a graph-to-value regression model to predict graph from scalar or vectors. Given $\{(s_1,[X_1]), \dots, (s_k, [X_k])\}, (s_i,[X_i]) \in \mathbb{R}^p\times X/T $ we are searching for:
# $$f: \mathbb{R}^p\rightarrow X/T$$
# where $f\in \mathcal{F}(X/T)$ is a generalized geodesic regression model, i.e., the canonical projection onto Graph Space of a regression line $h_\beta : \mathbb{R}^p\rightarrow X$ of the form $$h_\beta(s) = \sum_{j=1}^{p} \beta_i s_i$$
# The AAC algorithm for regression combines the estimation of $h_\beta$ given $\{X_1, \dots, X_k\}, X_i \in X$
# $$\sum_{i=0}^{k} d_X(h_\beta(s_i), X_i)$$
# and the searching for $\{X_1, \dots, X_k\}, X_i \in X$ optimally aligned with respect to the prediction along the current regression model:
# $$\min_{t\in T}d_X(h_\beta(s_i),t^TX_it)$$

# In[12]:


total_space = GraphSpace(n_nodes=5)
total_space.equip_with_group_action()
total_space.equip_with_quotient_structure();


# In[13]:


s = gs.array([random.randint(0, 10) for i in range(10)])


# In[14]:


aac_reg = AAC(space=total_space, estimate="regression")


# In[15]:


aac_reg.fit(s, graphset_1);


# The coefficients are saved in the following attributes and they can be changed into a graph shape.

# In[16]:


aac_reg.total_space_estimator.coef_


# A graph can be predicted using the fit model and the corresponding prediction error can be computed:

# In[17]:


graph_pred = aac_reg.total_space_estimator.predict(s)

gs.sum(graph_space.metric.dist(graphset_1, graph_pred))

