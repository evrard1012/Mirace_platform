import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster

## Implémentation de la distance euclidienne
def euclidean_distance(ts1, ts2):
    return np.sqrt(np.sum((ts1 - ts2) ** 2))

## Implémentation de DTW avec la distance euclidienne
def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean_distance(s1[i - 1], s2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],    # insertion
                dtw_matrix[i, j - 1],    # suppression
                dtw_matrix[i - 1, j - 1] # correspondance
            )
    return dtw_matrix[n, m]

def create_segments(df, nb_years):
    years = sorted(df.columns.astype(int))  # Assurez-vous que les années sont triées
    segments = []
    
    for i in range(0, len(years), nb_years):  # Parcourir les années par pas de 4
        start_year = years[i]
        end_year = years[min(i + nb_years - 1, len(years) - 1)]  # S'assurer de ne pas dépasser la dernière année
        segments.append((start_year, end_year))
    
    return segments

## Calcul de la matrice de distance DTW
def compute_dtw_distance_matrix(data):
    n = data.shape[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = dtw_distance(data[i], data[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix

## Analyse CAH et création du GIF
def perform_cah(df, nb_years):

    data_segments = []
    Z_segments = []
    titles = []
    
    segments = create_segments(df, nb_years)
    countries = df.index
    image_files = []

    if not os.path.exists('dendrograms'):
        os.makedirs('dendrograms')

    for (start, end) in segments:
        segment_data = df.loc[:, [start, end]].values
        dist_matrix = compute_dtw_distance_matrix(segment_data)

        
        condensed_dist_matrix = squareform(dist_matrix) 
        
        data_segments.append(dist_matrix) # list of data segments
        linked = linkage(condensed_dist_matrix, method='ward')
        Z_segments.append(linked) # list of linked
        titles.append(f'{start}-{end}') # list of titles
        
        plt.figure(figsize=(12, 6))
        dendrogram(linked, labels=countries.to_list())
        plt.title(f'Dendrogramme CAH ({start}-{end})')
        plt.xlabel('Pays')
        plt.ylabel('Distance')
        plt.xticks(rotation=45, ha='right')  # Rotation des labels pour une meilleure lisibilité

        filename = f'dendrograms/dendrogram_{start}_{end}.png'
        plt.savefig(filename)
        image_files.append(filename)
        plt.close()

    # Création du GIF
    images = [imageio.imread(file) for file in image_files]
    imageio.mimsave('dendrogram_evolution.gif', images, duration=3)
    
    return data_segments, Z_segments, titles ,countries

# Fonction pour créer un DataFrame avec les groupes pour chaque pays et chaque segment
def create_clusters_dataframe(data_segments, Z_segments, countries, titles):
    """
    Crée un DataFrame indiquant le groupe auquel chaque pays appartient pour chaque segment.

    Args:
        data_segments: Liste des données brutes pour chaque segment.
        Z_segments: Liste des matrices de fusion pour chaque segment.
        countries: Liste des noms des pays.

    Returns:
        DataFrame contenant les groupes pour chaque pays et chaque segment.
    """
    # Initialiser le DataFrame
    df = pd.DataFrame(index=countries)
    
    for i, (data, Z) in enumerate(zip(data_segments, Z_segments)):
        optimal_k = determine_optimal_clusters(Z, data, max_clusters=5)
        clusters = fcluster(Z, t=optimal_k, criterion='maxclust')
        #print(i, clusters)
        
        # Ajouter une colonne pour le segment courant
        df[f'Segment {i+1}'] = clusters

    df.columns = titles
    return df

def determine_optimal_clusters(Z, data, max_clusters=5):
    """
    Détermine le nombre optimal de clusters en utilisant la méthode du coude.

    Args:
        Z: Matrice de fusion générée par linkage.
        data: Données brutes.
        max_clusters: Nombre maximal de clusters à considérer.

    Returns:
        Nombre optimal de clusters.
    """
    inertias = []
    
    for k in range(2, max_clusters + 1):
        clusters = fcluster(Z, t=k, criterion='maxclust')
        inertia = sum([np.sum((data[clusters == i] - np.mean(data[clusters == i], axis=0))**2) for i in range(1, k+1)])
        inertias.append(inertia)
    
    # Trouver le point d'inflexion (méthode du coude)
    diff = np.diff(inertias)
    optimal_k = np.argmax(diff) + 2  # +2 car on commence à 2 clusters
    
    return optimal_k

def dataset_profile1(file_name, sheet_name=0, index_col=0, header=0, nb_years=5):
    import pandas as pd

    # --- Chargement du fichier ---
    if str(file_name).lower().endswith(".csv"):
        df = pd.read_csv(file_name, index_col=index_col, header=header)
    else:
        df = pd.read_excel(
            file_name,
            index_col=index_col,
            sheet_name=sheet_name,
            engine="openpyxl",
            header=header,
        )

    # --- Nettoyage des colonnes : garder uniquement les années ---
    # → Si la première colonne s'appelle 'Pays', elle est déjà index_col=0
    # → Les colonnes restantes doivent être des années : 1991, ..., 2023

    year_cols = [c for c in df.columns if str(c).strip().isdigit()]
    if not year_cols:
        raise ValueError("Aucune colonne d'année valide détectée. Vérifiez votre feuille.")

    df_years = df[year_cols].copy()
    df_years.columns = df_years.columns.astype(int)  # pour tri + cohérence

    # --- Lancement de l’analyse CAH sur les années uniquement ---
    data_segments, Z_segments, titles, countries = perform_cah(df_years, nb_years)

    # --- Clustering final regroupé par segment ---
    clusters_df = create_clusters_dataframe(data_segments, Z_segments, countries, titles)
    return clusters_df
