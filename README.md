## Spotify Music Analysis using Dimensionality Reduction and Clustering

### Project Overview

This project explores the Spotify Tracks Dataset (1921–2020) to discover hidden patterns in music using unsupervised machine learning, dimensionality reduction, and cluster analysis.

The objective is to investigate whether Spotify's audio features can reveal meaningful structures related to musical style, popularity, and historical evolution. By combining visualization techniques with clustering algorithms, the project demonstrates how machine learning can uncover insights from large, high-dimensional datasets.

## Project Objectives

The project aims to answer the following questions:

Can audio features naturally group songs into meaningful clusters?
How have musical characteristics evolved over the last 100 years?
Which audio features are most strongly associated with popularity?
Can genres be distinguished using Spotify's audio features alone?
Which dimensionality reduction techniques provide the most informative visualizations?

## Dataset

The analysis uses the Spotify Tracks Dataset (1921–2020) containing over 586,000 songs collected from Spotify's Web API.

Dataset Source

Kaggle: Spotify Dataset (1921–2020)

https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks

### Dataset Summary

Dataset Size
Tracks 586,672 songs
Artists 1,000,000+ artists
Audio Features 20
Time Span 1921–2020

### Audio Features

Examples include:

Danceability
Energy
Loudness
Tempo
Acousticness
Instrumentalness
Speechiness
Valence
Popularity

## Technologies Used

Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
PCA
t-SNE
UMAP
MDS

## Methodology

### 1. Data Preprocessing

Data cleaning
Handling missing values
Standardizing numerical features using StandardScaler
Converting release dates into numerical timestamps

### 2. Exploratory Data Analysis

Initial exploration focused on understanding:

Feature distributions
Correlations
Temporal trends
Popularity statistics

### 3. Dimensionality Reduction

To visualize high-dimensional music data, several projection techniques were evaluated:

PCA
t-SNE
UMAP
MDS

Projection quality was assessed using:

Trustworthiness
Continuity
k-Nearest Neighbor overlap
Classification accuracy

### 4. Cluster Analysis

Several clustering algorithms were compared:

K-Means
DBSCAN

The optimal clustering strategy was selected using:

Silhouette Score
Cluster Balance
Average Cluster Size
Additional internal validation metrics

### 5. Feature Importance

To better understand each cluster, feature distributions were compared against the overall dataset.

Visualization methods included:

Histograms
Boxplots
Violin plots

The most discriminative audio features were identified using statistical ranking methods.

## Results

The analysis revealed several interesting patterns.

### Distinct Musical Clusters

t-SNE produced well-separated clusters representing different musical characteristics, particularly for:

Instrumental tracks
Speech-heavy tracks
Live recordings

### Evolution of Music

Songs from different decades occupy different regions in the projection space, indicating that musical characteristics have changed substantially over the past century.

### Popularity Trends

Popular songs tend to cluster within specific regions of the feature space, suggesting that certain combinations of audio features are associated with commercial success.

### Genre Classification

Genres showed considerable overlap.

This suggests that Spotify audio features alone are not sufficient to distinguish genres, highlighting the importance of additional information such as lyrics, cultural context, or artist identity.

## Future Improvements

Apply deep learning techniques such as Autoencoders for representation learning.
Explore graph-based analysis using artist similarity networks.
Develop an interactive dashboard for music exploration.
Incorporate lyrical features and sentiment analysis.
Build a recommendation system based on learned embeddings.

## How to Run
```shell
git clone git@github.com:malak315/visual_data_exploration.git
cd visual_data_exploration
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Key Skills Demonstrated

Data Cleaning
Exploratory Data Analysis
Data Visualization
Machine Learning
Unsupervised Learning
Dimensionality Reduction
Clustering
Statistical Analysis
Python Programming
Scientific Data Interpretation
