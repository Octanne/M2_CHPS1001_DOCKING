import os
import re
from tqdm import tqdm

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

def filter_molecule(molecule, mol_files):
    mol_atoms = list() # We store the atoms of the molecule that or interesting
    # We read each file for the molecule
    for file in mol_files:
        with open(f"{directory_ligand}/{file}", 'r') as f:
            lines = f.readlines()
            modelNum = 0
            energy = 1
            for line in lines:
                if line.startswith("DOCKED: MODEL"):
                    modelNum = line.split()[2]
                if line.startswith("DOCKED: USER    Estimated Free Energy of Binding"):
                    energy = float(line.split()[8])
                if energy < 0 and line.startswith("DOCKED: ATOM"):
                    mol_atoms.append(line)
    return mol_atoms

def calculate_COM_atom(atom):
    atom_data = atom.split()
    atom_x = atom_data[7]
    atom_y = atom_data[8]
    atom_z = atom_data[9]
    #print(f"Atom : {atom_x} {atom_y} {atom_z}")
    return [float(atom_x), float(atom_y), float(atom_z)]

def calculate_COM_cluster(cluster):
    com = [0, 0, 0]
    for atom in cluster:
        atom_com = calculate_COM_atom(atom)
        com[0] += atom_com[0]
        com[1] += atom_com[1]
        com[2] += atom_com[2]
    com[0] /= len(cluster)
    com[1] /= len(cluster)
    com[2] /= len(cluster)
    return com
    
def clustering_molecule(mol_atoms):
    
    # We create a list of clusters
    clusters = list()
    # We create a list of center of mass for each cluster
    clusters_com = list()
    
    i = 0
    for atom in tqdm(mol_atoms):
        # We get the atom center of mass
        atom_com = calculate_COM_atom(atom)
        # We get the cluster center of mass
        iCluster = 0
        find_cluster = False
        for cluster_com in clusters_com:
            # We calculate the distance between the atom and the cluster center of mass
            distance = ((atom_com[0] - cluster_com[0])**2 + (atom_com[1] - cluster_com[1])**2 + (atom_com[2] - cluster_com[2])**2)**0.5
            
            if distance < 10:
                clusters[iCluster].append(atom)
                # We update the cluster center of mass
                clusters_com[iCluster] = calculate_COM_cluster(clusters[iCluster])
                find_cluster = True
                break
            iCluster += 1
        # If the atom is not in a cluster, we create a new cluster
        if not find_cluster:
            clusters.append(list())
            clusters_com.append(atom_com)
            clusters[iCluster].append(atom)
            #print(f"New cluster {iCluster} started with atom {i} and center of mass {atom_com}")
        i += 1
    
    return clusters, clusters_com

def show_graphs_clusters(clusters, clusters_com, total_atoms):
    # We show the graph of the clusters
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    """fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for cluster in clusters:
        x = list()
        y = list()
        z = list()
        for atom in cluster:
            atom_data = atom.split()
            x.append(float(atom_data[7]))
            y.append(float(atom_data[8]))
            z.append(float(atom_data[9]))
        ax.scatter(x, y, z)
    # Show non bloquant
    plt.show(block=False)"""
    # We show the graph by color depending of the percentage in each cluster (cloud of points)
    """fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    i = 0
    for cluster in clusters:
        x = list()
        y = list()
        z = list()
        for atom in cluster:
            atom_data = atom.split()
            x.append(float(atom_data[7]))
            y.append(float(atom_data[8]))
            z.append(float(atom_data[9]))
        percentage = len(cluster) / total_atoms
        # More the percentage is high, more the color is red
        color = (1, 0, 0, percentage)
        
        ax.scatter(x, y, z, c=[color])
    # Show non bloquant
    plt.show(block=False)"""
    # We show the graph by color depending of the percentage in each cluster (Surface of the cluster)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    i = 0
    for cluster in clusters:
        x = list()
        y = list()
        z = list()
        # Filter cluster with less than 3 atoms
        if len(cluster) < 3:
            continue
        for atom in cluster:
            atom_data = atom.split()
            x.append(float(atom_data[7]))
            y.append(float(atom_data[8]))
            z.append(float(atom_data[9]))
        percentage = len(cluster) / total_atoms
        # More the percentage is high, more the color is red
        color = (1, 0, 0, percentage)
        
        ax.plot_trisurf(x, y, z, color=color)
    # Show non bloquant
    plt.show(block=False)
    # We show the graph by color depending of the percentage in each cluster (Only center of mass of the cluster)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    i = 0
    for cluster_com in clusters_com:
        x = cluster_com[0]
        y = cluster_com[1]
        z = cluster_com[2]
        percentage = len(clusters[i]) / total_atoms
        # More the percentage is high, more the color is red
        color = (1, 0, 0, percentage)
        
        ax.scatter(x, y, z, c=[color])
        i += 1
    # Show non bloquant
    plt.show(block=True)
    
def show_tables_clusters(clusters, clusters_com, total_atoms):
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
        print(f"Percentage : {percentage:03.4f}%")
        i += 1

# We list all ligands
folder_ligands = [ "galactose", "lactose", "minoxidil", "nebivolol", "resveratrol" ]
folder_ligands = [ "galactose" ]
# We list all proteins / ligands file docking sort by ligands
for ligand in folder_ligands:
    directory_ligand = f'data/Results_{ligand}'
    print("===================================")
    print(f"\nLigand : {ligand}")
    parsed_data = parse_files(directory_ligand)
    # We process each molecule
    for molecule in parsed_data:
        mol_files = parsed_data[molecule]
        mol_atoms = filter_molecule(molecule, mol_files)
        # We do now from the mol_atoms list the clustering
        print(f"\nMolecule {molecule} :")
        print(f"Nb of atoms : {len(mol_atoms)}")
        mol_clustering = clustering_molecule(mol_atoms)
        clusters, clusters_com = mol_clustering
        print(f"Nb of clusters : {len(clusters)}")
        # Show graphs and tables of statistics of clusters repartition
        show_tables_clusters(clusters, clusters_com, len(mol_atoms))
        show_graphs_clusters(clusters, clusters_com, len(mol_atoms))
        break
    print("===================================")
    

## Filtration d'abord conserver juste les fichiers avec le meilleur score (négatif car viable)
## Faire parcours des fichiers 