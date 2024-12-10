import os
import re
from tqdm import tqdm

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from multiprocessing import Pool
import multiprocessing as mp
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import time

POINT_SPACING=1 # Point spacing in Angstroms

# Définir le mode de démarrage multiprocessing
def set_spawn_method():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# Parsing des fichiers
def parse_files(directory):
    pattern = re.compile(r'(\d+)_(\d+)_(\d+)_(\w+)\.dlg')
    result = {}
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            _, _, _, molecule_name = match.groups()
            result.setdefault(molecule_name, []).append(filename)
    return result

# Filtrage des molécules par score énergétique
def filter_molecule(mol_files, directory_ligand):
    filtered_atoms = []
    for file in mol_files:
        with open(f"{directory_ligand}/{file}", 'r') as f:
            lines = f.readlines()
            energy = float('inf')
            for line in lines:
                if line.startswith("DOCKED: USER    Estimated Free Energy of Binding"):
                    energy = float(line.split()[8])
                if energy < 0 and line.startswith("DOCKED: ATOM"):
                    filtered_atoms.append(line)
    return filtered_atoms

# Calcul GPU des coordonnées
def atom_to_tensor(atom):
    data = atom.split()
    return torch.tensor([float(data[7])*POINT_SPACING, float(data[8])*POINT_SPACING, float(data[9])*POINT_SPACING], device='cuda')

# Clustering basé sur DBSCAN avec GPU
def clustering_molecule_gpu(mol_atoms):
    if not mol_atoms:
        return [], []
    
    coords = torch.stack([atom_to_tensor(atom) for atom in mol_atoms]).cpu().numpy()
    db = DBSCAN(eps=10, min_samples=3).fit(coords)
    
    print(f"Clustering of {len(coords)} atoms in {len(set(db.labels_))} clusters")
    
    clusters = []
    for cluster_id in set(db.labels_):
        if cluster_id != -1:  # Ignorer les bruits
            clusters.append(coords[db.labels_ == cluster_id])
    
    cluster_centers = [np.mean(cluster, axis=0) for cluster in clusters]
    return clusters, cluster_centers

def show_graphs_clusters(molecule, clusters, clusters_com, total_atoms):  
    # We show the graph by color depending of the percentage in each cluster (Only center of mass of the cluster)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = get_cmap('coolwarm')  # Palette de couleurs du bleu au rouge
    # Calculer les pourcentages
    percentages = [len(cluster) / total_atoms for cluster in clusters]
    # Normalisation dynamique basée sur les pourcentages
    norm = Normalize(vmin=min(percentages), vmax=max(percentages))
    for i, cluster_com in enumerate(clusters_com):
        x = cluster_com[0]
        y = cluster_com[1]
        z = cluster_com[2]
        percentage = percentages[i]  # Pourcentage du cluster actuel
        # Obtenir la couleur normalisée
        color = cmap(norm(percentage))  # Appliquer la colormap à la valeur normalisée
        ax.scatter(x, y, z, color=[color], s=100)  # Ajout du point avec la couleur
    ax.set_title(f"({molecule}) Clusters Center of Mass (Color-coded by Percentage)")
    #plt.show(block=False)
    # Save the graph in a file in results folder
    fig.savefig(f"results/{ligand}_{molecule}_clusters_com.png")
    
    # We show the graph by color depending of the percentage in each cluster (Remove abnormal values)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = get_cmap('coolwarm')  # Palette de couleurs du bleu au rouge
    # Calculer les pourcentages
    percentages = [len(cluster) / total_atoms for cluster in clusters if len(cluster) >= 3]
    # Normalisation dynamique basée sur les pourcentages
    norm = Normalize(vmin=min(percentages), vmax=max(percentages))
    for cluster in clusters:
        x = []
        y = []
        z = []
        # Filtrer les clusters avec moins de 3 atomes
        if len(cluster) < 3:
            continue
        for atom in cluster:
            x.append(atom[0])
            y.append(atom[1])
            z.append(atom[2])        
        z_threshold = 2
        # Nettoyage des valeurs aberrantes
        x, y, z = np.array(x), np.array(y), np.array(z)
        z_mean, z_std = np.mean(z), np.std(z)
        mask = np.abs(z - z_mean) <= z_threshold * z_std  # Filtre basé sur l'écart-type
        x, y, z = x[mask], y[mask], z[mask]
        if len(x) < 3:  # Vérifie si les points restants permettent de tracer une surface
            continue
        percentage = len(cluster) / total_atoms  # Pourcentage du cluster actuel
        color = cmap(norm(percentage))  # Obtenir la couleur normalisée
        # Dessiner la surface triangulée avec la couleur normalisée
        ax.plot_trisurf(x, y, z, color=color, alpha=0.8)
    ax.set_title(f"({molecule}) Cluster Surfaces (Cleaned and Color-coded by Percentage)")
    #plt.show(block=True)
    # Save the graph in a file in results folder
    fig.savefig(f"results/{ligand}_{molecule}_clusters.png")
    # We close the graph
    plt.close('all')

def show_tables_clusters(molecule, clusters, clusters_com, total_atoms):
    # We show the tables of the clusters
    i = 0
    total_percentage = 0
    for cluster in clusters:
        print(f"Cluster {i:04d} :", end=' | ')
        print(f"Nb of atoms : {len(cluster):04d}", end=' | ')
        com_cluster = f"({clusters_com[i][0]:+08.2f} {clusters_com[i][1]:+08.2f} {clusters_com[i][2]:+08.2f})"
        print(f"Center of mass : {com_cluster}", end=' | ')
        percentage = len(cluster) / total_atoms * 100
        total_percentage += percentage
        print(f"Percentage : {percentage:08.4f}%")
        i += 1
    # Save the data in a file in results folder
    with open(f"results/{ligand}_{molecule}_results.txt", 'w') as f:
        f.write(f"Molecule : {molecule}\n")
        f.write(f"Nb of atoms : {total_atoms}\n")
        f.write(f"Nb of clusters : {len(clusters)}\n")
        f.write(f"Total percentage : {total_percentage:03.4f}%\n")
        i = 0
        f.write("| Cluster ID | Nb of atoms | Center of mass | Percentage |\n")
        for cluster in clusters:
            f.write(f"| Cluster {i:04d} |")
            f.write(f" {len(cluster):04d} |")
            com_cluster = f"({clusters_com[i][0]:+08.2f} {clusters_com[i][1]:+08.2f} {clusters_com[i][2]:+08.2f})"
            f.write(f" {com_cluster} |")
            percentage = len(cluster) / total_atoms * 100
            f.write(f" {percentage:08.4f}% |\n")
            i += 1

# Traitement d'une molécule
def process_molecule(args):
    molecule, parsed_data, directory_ligand = args
    mol_files = parsed_data[molecule]
    mol_atoms = filter_molecule(mol_files, directory_ligand)
    clusters, centers = clustering_molecule_gpu(mol_atoms)
    return molecule, clusters, centers, len(mol_atoms)

# We save the time in a file
time_tabs = dict()
start_time = 0
def check_time(check_pt):
    global start_time
    if start_time == 0:
        start_time = time.time()
    else:
        end_time = time.time()
        time_tabs[f"{len(time_tabs)}-"+check_pt] = end_time - start_time
        start_time = end_time
def save_time():
    with open("results/time_cpu.txt", 'w') as f:
        for time in time_tabs:
            f.write(f"{time} : {time_tabs[time]}\n")

# Main
if __name__ == "__main__":
    set_spawn_method()
    folder_ligands = ["galactose", "lactose", "minoxidil", "nebivolol", "resveratrol"]
    
    for ligand in folder_ligands:
        check_time("start-"+ligand)
        directory_ligand = f"data/Results_{ligand}"
        parsed_data = parse_files(directory_ligand)
        print("====================================")
        print(f"Processing ligand {ligand}...")

        tasks = [(molecule, parsed_data, directory_ligand) for molecule in parsed_data]
        
        with Pool(processes=max(1, mp.cpu_count() // 4)) as pool:
            results = pool.map(process_molecule, tasks)

        for molecule, clusters, centers, total_atoms in results:
            show_tables_clusters(molecule, clusters, centers, total_atoms)
            show_graphs_clusters(molecule, clusters, centers, total_atoms)
        check_time("end-"+ligand)
        print(f"End of processing ligand {ligand}!")
        print("====================================")
        save_time()
