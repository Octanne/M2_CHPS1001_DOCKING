import os
import re

from tqdm import tqdm
import matplotlib.pyplot as plt
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
RESULT_FOLDER="results/juliet/results_gpu_clus_100_atoms_2250"
CPU_COUNT=20
ANGSTROMS=10
NONE_CLUSTER = "[ None ]"

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
    return cp.array([float(atom_x)*POINT_SPACING, float(atom_y)*POINT_SPACING, float(atom_z)*POINT_SPACING], dtype=cp.float32)

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
    
def fusion_clusters_cpu(clusters, clusters_com):
    for i in clusters_com.keys():
        for j in clusters_com.keys():
            if i != j:
                if clusters[i] == NONE_CLUSTER or clusters[j] == NONE_CLUSTER:
                    continue
                dist = cp.linalg.norm(cp.array(clusters_com[i]) - cp.array(clusters_com[j]))
                if dist < ANGSTROMS:
                    clusters_com[i] = calculate_com_cluster_combine(clusters[i], clusters_com[i], clusters[j], clusters_com[j])
                    clusters[i] = clusters[i] + clusters[j]
                    clusters[j] = NONE_CLUSTER # We put the cluster to a value that is not possible
                    clusters_com[j] = NONE_CLUSTER # We put the cluster_com to a value that is not possible
                    
    check_time("final") # We start the time for the final clustering
    # We convert the clusters and clusters_com to the final list
    final_clusters = []
    final_clusters_com = []
    for i in clusters_com.keys():
        if clusters[i] != NONE_CLUSTER:
            final_clusters.append(clusters[i])
            final_clusters_com.append(clusters_com[i])
    check_time("final") # We save the time for the final clustering

    return final_clusters, final_clusters_com
    
def calc_section_gpu(args):
    if len(args) == 3:
        i, atoms_com_array, nb_atoms_per_section = args
        clusters_past = None
    else:
        i, atoms_com_array, nb_atoms_per_section, clusters_past = args
    
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
    
    #first=i*nb_atoms_per_section
    #last=(i+1)*nb_atoms_per_section-1
    #print("Section", i, f"cluster indices {first} to {last} :")
    
    #print(f"Section {i} nb_cluster on the section : {num_atoms}")
    #print(f"Section {i} cluster_indices : {cluster_indices}")
    
    clusters = dict()
    clusters_com = dict()
    
    # Process cluster indices to form clusters
    if clusters_past is None:
        for atom_id in tqdm(range(num_atoms)):
            atom_com = section[atom_id] # Il s'agit du centre de masse de l'atome
            atom_id_group = int(cluster_indices[atom_id]) # Il s'agit de l'indice de l'atome vers lequel l'atome pointe
            
            # On vérifie si lui même n'est pas utilisé par d'autres atomes
            if atom_id in clusters.keys():
                atom_id_group = atom_id
                # On l'update dans le dico des indices
                cluster_indices[atom_id] = atom_id_group
            # Pour faire repointer les atomes d'un cluster qui pointes vers un atome déjà dans un cluster
            elif atom_id_group not in clusters.keys():
                # Faire recursive qui remonte les atomes qui pointent vers un atome déjà dans un cluster si pas dans un cluster a la fin de la chaine
                # alors on le met dans un cluster de atom_id_group
                anti_loop = []
                potential_id_group = atom_id_group
                while int(cluster_indices[potential_id_group]) not in clusters.keys() and int(cluster_indices[potential_id_group]) not in anti_loop:
                    anti_loop.append(potential_id_group)
                    potential_id_group = int(cluster_indices[potential_id_group])
                if int(cluster_indices[potential_id_group]) in clusters.keys():
                    atom_id_group = int(cluster_indices[potential_id_group])
                    # On l'update dans le dico des indices
                    cluster_indices[atom_id] = atom_id_group
                else:
                    # On crée un nouveau cluster
                    clusters[atom_id_group] = [atom_com]
                    clusters_com[atom_id_group] = atom_com
                    continue
            
            # Si le cluster existe on ajoute l'atome au cluster
            clusters_com[atom_id_group] = calculate_com_cluster(clusters[atom_id_group], clusters_com[atom_id_group], atom_com)
            clusters[atom_id_group].append(atom_com)
    else:
        for cluster_id in tqdm(range(num_atoms)):
            cluster_com = section[cluster_id] # Il s'agit du centre de masse du cluster
            cluster_id_group = int(cluster_indices[cluster_id]) # Il s'agit de l'indice du cluster vers lequel le cluster pointe
            
            # On vérifie si lui même n'est pas utilisé par d'autres atomes
            if cluster_id in clusters.keys():
                cluster_id_group = cluster_id
                # On l'update dans le dico des indices
                cluster_indices[cluster_id] = cluster_id_group
            # Pour faire repointer les atomes d'un cluster qui pointes vers un atome déjà dans un cluster
            elif cluster_id_group not in clusters.keys():
                # Faire recursive qui remonte les atomes qui pointent vers un atome déjà dans un cluster si pas dans un cluster a la fin de la chaine
                # alors on le met dans un cluster de cluster_id_group
                anti_loop = []
                potential_id_group = cluster_id_group
                while int(cluster_indices[potential_id_group]) not in clusters.keys() and int(cluster_indices[potential_id_group]) not in anti_loop:
                    anti_loop.append(potential_id_group)
                    potential_id_group = int(cluster_indices[potential_id_group])
                if int(cluster_indices[potential_id_group]) in clusters.keys():
                    cluster_id_group = int(cluster_indices[potential_id_group])
                    # On l'update dans le dico des indices
                    cluster_indices[cluster_id] = cluster_id_group
                else:
                    clusters[cluster_id_group] = clusters_past[cluster_id+i*nb_atoms_per_section]
                    clusters_com[cluster_id_group] = cluster_com
                    continue
            
            # Si le cluster existe on ajoute le cluster au cluster
            clusters_com[cluster_id_group] = calculate_com_cluster_combine(clusters[cluster_id_group], clusters_com[cluster_id_group], clusters_past[cluster_id+i*nb_atoms_per_section], cluster_com)
            clusters[cluster_id_group].extend(clusters_past[cluster_id+i*nb_atoms_per_section])
    
    return clusters, clusters_com

def prepare_data_atoms(mol_atoms):
    check_time("calc_com_atoms") # We start the time for the center of mass calculation
    atoms_com = [calculate_com_atom(atom) for atom in mol_atoms]
    check_time("calc_com_atoms") # We save the time for the center of mass calculation
    
    check_time("sort_atoms") # We start the time for the sorting
    atoms_com_array = cp.array(atoms_com, dtype=cp.float32)  # Move atoms_com to GPU
    # We sort the atoms by the z axis then the y axis then the x axis
    atoms_com_array = atoms_com_array[cp.lexsort(cp.stack((atoms_com_array[:, 0], atoms_com_array[:, 1], atoms_com_array[:, 2])))]
    check_time("sort_atoms") # We save the time for the sorting
    
    check_time("calc_sections_atoms") # We start the time for the copy to device
    nb_atoms_per_section = 2250
    num_sections = (len(atoms_com_array) + (nb_atoms_per_section-1)) // nb_atoms_per_section 
    check_time("calc_sections_atoms") # We save the time for the copy to device
    
    return num_sections, atoms_com_array, nb_atoms_per_section

def prepare_data_clusters(glusters_com):
    check_time("calc_com_glusters") # We start the time for the center of mass calculation
    glusters_com = [gluster for gluster in glusters_com]
    check_time("calc_com_glusters") # We save the time for the center of mass calculation
    
    check_time("sort_glusters") # We start the time for the sorting
    glusters_com_array = cp.array(glusters_com, dtype=cp.float32)  # Move atoms_com to GPU
    #print(f"Clusters com array : {glusters_com_array}")
    # We sort the atoms by the z axis then the y axis then the x axis
    #clusters_sort_index = cp.lexsort(cp.stack((glusters_com_array[:, 0], glusters_com_array[:, 1], glusters_com_array[:, 2])))
    #glusters_com_array = glusters_com_array[clusters_sort_index]
    #print(f"Clusters com array sorted : {glusters_com_array}")
    #print(f"Clusters sort index : {clusters_sort_index}")
    check_time("sort_glusters") # We save the time for the sorting
    
    check_time("calc_sections_glusters") # We start the time for the copy to device
    nb_gl_per_section = 100
    num_sections = (len(glusters_com_array) + (nb_gl_per_section-1)) // nb_gl_per_section 
    check_time("calc_sections_glusters") # We save the time for the copy to device
    
    return num_sections, glusters_com_array, nb_gl_per_section

def check_nb_of_atoms(clusters, supposed_nb_atoms):
    nb_total_atoms = 0
    for i in clusters:
        nb_total_atoms += len(i)
    assert nb_total_atoms == supposed_nb_atoms, f"Nb of atoms : {nb_total_atoms} != Nb of supposed atoms : {supposed_nb_atoms}"

def clustering_molecule(mol_atoms): 
    check_time("clustering") # We start the time for the clustering
    
    check_time("preparation") # We start the time
    num_sections, atoms_com_array, nb_atoms_per_section = prepare_data_atoms(mol_atoms)
    check_time("preparation") # We save the time

    check_time("sect_atoms") # We start the time for the section clustering
    pool = Pool(min(CPU_COUNT, num_sections))
    results = pool.map(calc_section_gpu, [(i, atoms_com_array, nb_atoms_per_section) for i in range(num_sections)])
    # get the results from the pool and fusion the dict
    clusters = []
    clusters_com = []
    for result in results:
        clusters.extend(result[0].values())
        clusters_com.extend(result[1].values())
    check_time("sect_atoms") # We save the time for the section clustering
    
    check_nb_of_atoms(clusters, len(atoms_com_array)) # Check if we have the same number of atoms
    
    #print("Nb of clusters before fusion : ", len(clusters))
    
    # We fusion the clusters that are at less than 10 Angstroms from each other by using a kernel
    _="""clusters, clusters_com = fusion_clusters_cpu(clusters, clusters_com)"""
    check_time("sect_clusters") # We start the time for the section clustering
    num_sec_gl, gl_com_array, nb_gl_per_section = prepare_data_clusters(clusters_com)
    pool = Pool(min(CPU_COUNT, num_sec_gl))
    results = pool.map(calc_section_gpu, [(i, gl_com_array, nb_gl_per_section, clusters) for i in range(num_sec_gl)])
    
    # get the results from the pool and fusion the dict
    clusters = []
    clusters_com = []
    for result in results:
        clusters.extend(result[0].values())
        clusters_com.extend(result[1].values())
    
    check_nb_of_atoms(clusters, len(atoms_com_array)) # Check if we have the same number of atoms
    check_time("sect_clusters") # We save the time for the section clustering
    
    check_time("clustering") # We save the time for the clustering
    save_time()
    
    return clusters, clusters_com

# We process the list of atoms to create the clusters in GPU by sections of 100 atoms (to avoid memory overflow)
# We do the clustering by sections of 100 atoms in parallel with the GPU (CUDA) and we merge the results in the CPU
# So we need to create the sections of 100 atoms and the last section can have less than 100 atoms
# Each section is a list of atoms_com and we have to make run a kernel on each section to get the clusters
# We have to merge the clusters of each section to get the final clusters
# We have to merge the clusters_com of each section to get the final clusters_com
# We have to merge the total number of atoms to get the total number of atoms
# We check at the end that we have the same number of atoms in the clusters and the total number of atoms

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
    # Save the data in a file in results folder
    with open(f"{RESULT_FOLDER}/{ligand}_{molecule}_results.txt", 'w') as f:
        f.write(f"Molecule : {molecule}\n")
        f.write(f"Nb of atoms : {total_atoms}\n")
        f.write(f"Nb of clusters : {len(clusters)}\n")
        i = 0
        total_percentage = sum([len(cluster) / total_atoms * 100 for cluster in clusters])
        f.write(f"Total percentage : {total_percentage:03.4f}%\n")
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