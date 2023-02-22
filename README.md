# split-DBSCAN

[![Tests](https://github.com/tadasnik/split-DBSCAN/actions/workflows/python-app.yml/badge.svg)](https://github.com/tadasnik/split-DBSCAN/actions/workflows/python-app.yml)

A class extending scikit-learn DBSCAN to allow the clustering to be performed in chunks and merge the clusters.

It will only work if the input dataset is sorted along the chunking dimension and if there is reasonable separation between the clusters (they do not extend all the way along the chunking dimension).  
