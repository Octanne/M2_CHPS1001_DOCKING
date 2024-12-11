import os
import re

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import numpy as np
import cupy as cp

from multiprocessing import Pool
import multiprocessing as mp
import time

# We need to set in spawn mode to be able to use the multiprocessing with CUDA
def set_start_method():
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

POINT_SPACING=0.375 # Point spacing in Angstroms
RESULT_FOLDER="results/results_gpu"
CPU_COUNT=8

def parse_files(directory):
    result = dict()
    pattern = re.compile(r'(\d+)_(\d+)_(\d+)_(\w+)\.dlg')

    # We iterate over all files in the directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            _, _, _, molecule_name = match.groups()
            if result.get(molecule_name) == None:
                result[molecule_name] = list()
            result[molecule_name].append(filename)
    return result

def print_molecule(directory):
    parsed_data = parse_files(directory)
    molecule_nb = 1
    for data in parsed_data:
        print(f"Molecule {data} ({molecule_nb}) :")
        for file in parsed_data[data]:
            print(f"{file}")
        molecule_nb += 1

def filter_molecule(molecule, mol_files, directory_ligand):
    mol_atoms = list() # We store the atoms of the molecule that or interesting
    # We read each file for the molecule
    for file in mol_files:
        with open(f"{directory_ligand}/{file}", 'r') as f:
            lines = f.readlines()
            energy = 1
            for line in lines:
                #if line.startswith("DOCKED: MODEL"):
                    #model_num = line.split()[2]
                if line.startswith("DOCKED: USER    Estimated Free Energy of Binding"):
                    energy = float(line.split()[8])
                if energy < 0 and line.startswith("DOCKED: ATOM"):
                    mol_atoms.append(line)
    return mol_atoms

def calculate_com_atom(atom):
    atom_data = atom.split()
    atom_x = atom_data[7]
    atom_y = atom_data[8]
    atom_z = atom_data[9]
    return cp.array([float(atom_x), float(atom_y), float(atom_z)], dtype=cp.float32)

def calculate_com_cluster(cluster, cluster_com, atom_com):
    # We add atom_com to the mean of the cluster
    com = [0, 0, 0]
    com[0] = (cluster_com[0] * len(cluster) + atom_com[0]) / (len(cluster) + 1)
    com[1] = (cluster_com[1] * len(cluster) + atom_com[1]) / (len(cluster) + 1)
    com[2] = (cluster_com[2] * len(cluster) + atom_com[2]) / (len(cluster) + 1)
    return com

def calculate_com_cluster_combine(cluster1, cluster_com1, cluster2, cluster_com2):
    # We add atom_com to the mean of the cluster
    com = [0, 0, 0]
    com[0] = (cluster_com1[0] * len(cluster1) + cluster_com2[0] * len(cluster2)) / (len(cluster1) + len(cluster2))
    com[1] = (cluster_com1[1] * len(cluster1) + cluster_com2[1] * len(cluster2)) / (len(cluster1) + len(cluster2))
    com[2] = (cluster_com1[2] * len(cluster1) + cluster_com2[2] * len(cluster2)) / (len(cluster1) + len(cluster2))
    return com

# Kernel definition for atoms clustering 
cluster_atom_kernel = cp.ElementwiseKernel( 
    'raw float32 x, raw float32 y, raw float32 z, float32 threshold, int32 num_atoms',
    'int32 cluster_index', 
    ''' 
    int idx = i; 
    for (int j = 0; j < num_atoms; j++) { 
        if (j != idx) { 
            float dist = sqrt((x[idx] - x[j]) * (x[idx] - x[j]) 
                            + (y[idx] - y[j]) * (y[idx] - y[j]) 
                            + (z[idx] - z[j]) * (z[idx] - z[j]));
            if (dist < threshold) { 
                cluster_index = j; 
                return; 
            }
        } 
    } 
    cluster_index = idx; ''', 
    'cluster_atom_kernel' )

def clustering_molecule_sect(mol_atoms):    
    # We convert the list of atoms in a list of center of mass
    atoms_com = cp.array([calculate_com_atom(atom) for atom in mol_atoms], dtype=cp.float32)
    # We offload the atoms_com to the GPU
    atoms_com = cp.array(atoms_com)
    # We sort the atoms_com by the z axis then the y axis then the x axis
    sort_indices = cp.lexsort(cp.stack((atoms_com[:, 0], atoms_com[:, 1], atoms_com[:, 2])))
    atoms_com = atoms_com[sort_indices]
        
    # We define the number of atoms per section and the threshold for clustering
    nb_atoms = len(atoms_com)
    nb_atoms_per_section = 2500
    nb_sections = (len(atoms_com) + (nb_atoms_per_section-1)) // nb_atoms_per_section
    ANGSTROMS = 10.0
    
    # We create the list of clusters and clusters_com in the GPU
    clusters = cp.zeros((nb_atoms, 3), dtype=cp.float32, order='C')
    clusters_com = cp.zeros((nb_atoms, 3), dtype=cp.float32, order='C')
    
    print("cluster : ", clusters)
    
    # We flatten the atoms_com array to be able to process it in the kernel
    atoms_comX = cp.ascontiguousarray(atoms_com[:, 0].flatten())
    atoms_comY = cp.ascontiguousarray(atoms_com[:, 1].flatten())
    atoms_comZ = cp.ascontiguousarray(atoms_com[:, 2].flatten())
    
    print(f"Nb of atoms : {nb_atoms}")
    print(f"Nb of sections : {nb_sections}")
    print(f"atoms_comX : {atoms_comX}")
    print(f"atoms_comY : {atoms_comY}")
    print(f"atoms_comZ : {atoms_comZ}")
    
    threads_per_block = 256
    blocks_per_grid = (nb_sections + (threads_per_block - 1)) // threads_per_block
    print(f"Threads per block : {threads_per_block}")
    
    # We process the sections in parallel with the GPU (CUDA) and we merge the results in the CPU
    
    # We fusion the clusters that are at less than 10 Angstroms from each other by using a kernel
    # TODO : We have to do this in the GPU with a kernel to be faster and more efficient (to be done)
    NONE_CLUSTER = cp.zeros((0, 3), dtype=cp.float32)
    for i in range(len(clusters_com)):
        for j in range(len(clusters_com)):
            if i != j:
                if cp.array_equal(clusters[i], NONE_CLUSTER) or cp.array_equal(clusters[j], NONE_CLUSTER):
                    continue
                dist = cp.linalg.norm(clusters_com[i] - clusters_com[j])
                if dist < ANGSTROMS:
                    clusters[i] = cp.concatenate((clusters[i], clusters[j]), axis=0)
                    clusters_com[i] = calculate_com_cluster_combine(clusters[i], clusters_com[i], clusters[j], clusters_com[j])
                    clusters[j] = NONE_CLUSTER
                    clusters_com[j] = NONE_CLUSTER
    # We convert the clusters and clusters_com to the final list
    final_clusters = [cluster for cluster in clusters if not cp.array_equal(cluster, NONE_CLUSTER)]
    final_clusters_com = [cluster_com for cluster_com in clusters_com if not cp.array_equal(cluster_com, NONE_CLUSTER)]
    
    return final_clusters, final_clusters_com
    
def calc_section(args):
    i, atoms_com_array, nb_atoms_per_section, ANGSTROMS = args
    
    section = atoms_com_array[i*nb_atoms_per_section:(i+1)*nb_atoms_per_section] 
    num_atoms = section.shape[0]
    d_section = cp.array(section)
    
    # Extract x, y, z coordinates 
    x = d_section[:, 0] 
    y = d_section[:, 1] 
    z = d_section[:, 2] 
    
    # Initialize cluster indices 
    cluster_indices = cp.zeros(num_atoms, dtype=cp.int32)
    
    # Launch the kernel 
    cluster_atom_kernel(x, y, z, ANGSTROMS, num_atoms, cluster_indices)
    print("sector : ", i)
    
    clusters = dict()
    clusters_com = dict()
    
    # Process cluster indices to form clusters 
    for j in range(num_atoms):
        idx = int(cluster_indices[j]+i*nb_atoms_per_section) # We add the offset of the section
        atom_com = section[j]
        if idx not in clusters:
            clusters[idx] = [atom_com]
            clusters_com[idx] = atom_com
        else:
            clusters[idx].append(atom_com)
            clusters_com[idx] = calculate_com_cluster(clusters[idx], clusters_com[idx], atom_com)
            
    return clusters, clusters_com

def clustering_molecule(mol_atoms): 
    check_time("clustering") # We start the time for the clustering
    
    check_time("preparation") # We start the time
    
    check_time("calc_com") # We start the time for the center of mass calculation
    # TODO faire calc en parallèle
    atoms_com = [calculate_com_atom(atom) for atom in mol_atoms]
    check_time("calc_com") # We save the time for the center of mass calculation
    
    check_time("sort") # We start the time for the sorting
    atoms_com_array = cp.array(atoms_com, dtype=cp.float32)  # Move atoms_com to GPU
    # We sort the atoms by the z axis then the y axis then the x axis
    atoms_com_array = atoms_com_array[cp.lexsort(cp.stack((atoms_com_array[:, 0], atoms_com_array[:, 1], atoms_com_array[:, 2])))]
    check_time("sort") # We save the time for the sorting
    
    check_time("calc sections") # We start the time for the copy to device
    ANGSTROMS = 10.0
    nb_atoms_per_section = 4500
    num_sections = (len(atoms_com_array) + (nb_atoms_per_section-1)) // nb_atoms_per_section 
    check_time("calc sections") # We save the time for the copy to device
    
    check_time("preparation") # We save the time

    check_time("sect") # We start the time for the section clustering
    # TODO make a pool to calc section in parallel
    pool = Pool(min(CPU_COUNT, num_sections))
    results = pool.map(calc_section, [(i, atoms_com_array, nb_atoms_per_section, ANGSTROMS) for i in range(num_sections)])
    
    # get the results from the pool and fusion the dict
    clusters = dict()
    clusters_com = dict()
    for result in results:
        for key in result[0]:
            assert key not in clusters # Check if the key is not already in the clusters
            clusters[key] = result[0][key]
            clusters_com[key] = result[1][key]
    check_time("sect") # We save the time for the section clustering
    
    check_time("final") # We start the time for the final clustering
    final_clusters = []
    final_clusters_com = []
    
    nb_total_atoms = 0
    for i in clusters.keys():
        nb_total_atoms += len(clusters[i])
    assert nb_total_atoms == len(atoms_com) # Check if we have the same number of atoms
                
    # We fusion the clusters that are at less than 10 Angstroms from each other by using a kernel
    # TODO : We have to do this in the GPU with a kernel to be faster and more efficient (to be done)
    NONE_CLUSTER = "[ None ]"
    for i in clusters_com.keys():
        for j in clusters_com.keys():
            if i != j:
                if clusters[i] == NONE_CLUSTER or clusters[j] == NONE_CLUSTER:
                    continue
                dist = cp.linalg.norm(cp.array(clusters_com[i]) - cp.array(clusters_com[j]))
                if dist < ANGSTROMS:
                    clusters[i] = clusters[i] + clusters[j]
                    clusters_com[i] = calculate_com_cluster_combine(clusters[i], clusters_com[i], clusters[j], clusters_com[j])
                    clusters[j] = NONE_CLUSTER # We put the cluster to a value that is not possible
                    clusters_com[j] = NONE_CLUSTER # We put the cluster_com to a value that is not possible
    # We convert the clusters and clusters_com to the final list
    for i in clusters_com.keys():
        if clusters[i] != NONE_CLUSTER:
            final_clusters.append(clusters[i])
            final_clusters_com.append(clusters_com[i])
    check_time("final") # We save the time for the final clustering
    
    check_time("clustering") # We save the time for the clustering
    save_time()
    
    return final_clusters, final_clusters_com

oldcode="""    
def clustering_molecule(mol_atoms):
    # We create a list of clusters
    clusters = list()
    # We create a list of center of mass for each cluster
    clusters_com = list()
    # We convert the list of atoms in a list of center of mass
    atoms_com = [calculate_com_atom(atom) for atom in mol_atoms]
    
    # We sort the atoms by the z axis then the y axis then the x axis
    atoms_com.sort(key=lambda x: (x[2], x[1], x[0]))
    
    # We process the list of atoms to create the clusters in GPU by sections of 100 atoms (to avoid memory overflow)
    # We do the clustering by sections of 100 atoms in parallel with the GPU (CUDA) and we merge the results in the CPU
    # So we need to create the sections of 100 atoms and the last section can have less than 100 atoms
    # Each section is a list of atoms_com and we have to make run a kernel on each section to get the clusters
    # We have to merge the clusters of each section to get the final clusters
    # We have to merge the clusters_com of each section to get the final clusters_com
    # We have to merge the total number of atoms to get the total number of atoms
    # We check at the end that we have the same number of atoms in the clusters and the total number of atoms
    
    return clusters, clusters_com
"""

def show_graphs_clusters(molecule, clusters, clusters_com, total_atoms):
    # We show the graph by color depending of the percentage in each cluster (Only center of mass of the cluster)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('coolwarm')  # Palette de couleurs du bleu au rouge
    # Calculer les pourcentages
    percentages = [len(cluster) / total_atoms for cluster in clusters]
    # Normalisation dynamique basée sur les pourcentages
    norm = Normalize(vmin=min(percentages), vmax=max(percentages))
    for i, cluster_com in enumerate(clusters_com):
        x = float(cluster_com[0])
        y = float(cluster_com[1])
        z = float(cluster_com[2])
        percentage = percentages[i]  # Pourcentage du cluster actuel
        # Obtenir la couleur normalisée
        color = cmap(norm(percentage))  # Appliquer la colormap à la valeur normalisée
        ax.scatter(x, y, z, color=[color], s=100)  # Ajout du point avec la couleur
    ax.set_title(f"({molecule}) Clusters Center of Mass (Color-coded by Percentage)")
    #plt.show(block=False)
    # Save the graph in a file in results folder
    fig.savefig(f"{RESULT_FOLDER}/{ligand}_{molecule}_clusters_com.png")
    
    # We show the graph by color depending of the percentage in each cluster (Remove abnormal values)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('coolwarm')  # Palette de couleurs du bleu au rouge
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
            x.append(float(atom[0]))
            y.append(float(atom[1]))
            z.append(float(atom[2]))       
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
    fig.savefig(f"{RESULT_FOLDER}/{ligand}_{molecule}_clusters.png")
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
    with open(f"{RESULT_FOLDER}/{ligand}_{molecule}_results.txt", 'w') as f:
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

def process_molecule(args):
    molecule, parsed_data, directory_ligand, ligand = args
    check_time(f"{ligand}_{molecule}") # We start the time for the molecule
    mol_files = parsed_data[molecule]
    mol_atoms = filter_molecule(molecule, mol_files, directory_ligand)
    # We do now from the mol_atoms list the clustering
    mol_clustering = clustering_molecule(mol_atoms)
    mol_clustering = (molecule, mol_atoms, *mol_clustering)
    check_time(f"{ligand}_{molecule}") # We save the time for the molecule
    return mol_clustering

def print_cluster_info(mol_clustering):
    molecule, mol_atoms, clusters, clusters_com = mol_clustering    
    print(f"Molecule : {molecule}")
    print(f"Nb of atoms : {len(mol_atoms)}")
    print(f"Nb of clusters : {len(clusters)}")
    # Show graphs and tables of statistics of clusters repartition
    show_tables_clusters(molecule, clusters, clusters_com, len(mol_atoms))
    show_graphs_clusters(molecule, clusters, clusters_com, len(mol_atoms))

if __name__ == "__main__":
    set_start_method()
    
    # We save the time in a file
    time_tabs = dict()
    def check_time(check_pt):
        if check_pt in time_tabs.keys():
            time_tabs[check_pt] = time.time() - time_tabs[check_pt]
        else:
            time_tabs[check_pt] = time.time()
    def save_time():
        with open(f"{RESULT_FOLDER}/time_gpu.txt", 'w') as f:
            for time in time_tabs:
                f.write(f"{time} : {time_tabs[time]}\n")

    # We list all ligands
    folder_ligands = [ "galactose", "lactose", "minoxidil", "nebivolol", "resveratrol" ]

    # We create the results folder
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
    check_time("total") # We start the total time
    # We list all proteins / ligands file docking sort by ligands
    for ligand in folder_ligands:
        directory_ligand = f'data/Results_{ligand}'
        print("===================================")
        print(f"Ligand : {ligand}")
        check_time(ligand) # We start the time for the ligand
        
        parsed_data = parse_files(directory_ligand)
        
        # Prepare the data for the multiprocessing
        tasks = [ (molecule, parsed_data, directory_ligand, ligand) for molecule in parsed_data ]

        # Iterate to each molecule and process the clustering
        results_async = map(process_molecule, tasks)
            
        # Print the results
        results = list()
        for mol_clustering in results_async:
            print(f"Processing molecule {mol_clustering[0]} done !")
            results.append(mol_clustering)
        for result in results:
            print_cluster_info(result)
        print("===================================")
        check_time(ligand) # We save the time for the ligand
        save_time()
    check_time("total") # We save the total time
    save_time()
    ## Filtration d'abord conserver juste les fichiers avec le meilleur score (négatif car viable)
    ## Faire parcours des fichiers 