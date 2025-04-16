# Written by Arman Moussavi
# Last updated: 04-16-2025

import numpy as np
import re
import networkx as nx

# Function to read and parse data from a LAMMPS .data file with atom-style full
def parse_lammps_data(file_path):
    # Initialize empty lists to store atom, bond, and angle data
    atoms = []
    bonds = []
    angles = []

    # Open the LAMMPS .data file for reading
    with open(file_path, 'r') as file:
        # Read all lines from the file
        lines = file.readlines()
        
    # Flags to track the current section (Atoms, Bonds, Angles)
    atom_section = False
    bond_section = False
    angle_section = False

    # Dictionary to store box bounds (x, y, z dimensions)
    box_bounds = {}

    # Regular expressions to capture the box bounds for x, y, and z directions
    box_regex = {
        'x': re.compile(r'^([-\d\.eE]+)\s+([-\d\.eE]+)\s+xlo\s+xhi'),
        'y': re.compile(r'^([-\d\.eE]+)\s+([-\d\.eE]+)\s+ylo\s+yhi'),
        'z': re.compile(r'^([-\d\.eE]+)\s+([-\d\.eE]+)\s+zlo\s+zhi')
    }

    # Loop through all lines in the file to extract the box bounds
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        
        # Check for box bounds in the x, y, and z directions
        for axis in ['x', 'y', 'z']:
            match = box_regex[axis].match(line)
            if match:
                # Extract the low and high bounds for each axis
                lo, hi = map(float, match.groups())
                box_bounds[axis] = hi - lo  # Store the range (hi - lo)
                break  # Stop searching after finding a match for the axis

    # Loop through the file again to extract atoms, bonds, and angles
    for line in lines:
        # Check for section headers and switch flags accordingly
        if line.strip() == "Atoms # full":
            atom_section = True  # Start reading atoms section
            continue
        elif line.strip() == "Bonds":
            bond_section = True  # Start reading bonds section
            atom_section = False  # Stop reading atoms section
            continue
        elif line.strip() == "Angles": 
            angle_section = True  # Start reading angles section
            bond_section = False  # Stop reading bonds section
            atom_section = False  # Stop reading atoms section
            continue
        elif line.strip() == "Velocities":
            # Stop reading all sections when "Velocities" is encountered
            bond_section = False
            atom_section = False
            angle_section = False
        
        # Parse atoms section
        if atom_section and line.strip() and not line.startswith("#"):
            parts = line.split()  # Split line into individual parts
            atom_id = int(parts[0])  # Atom ID
            mol_id = int(parts[1])  # Molecule ID
            atom_type = int(parts[2])  # Atom type
            x, y, z = float(parts[4]), float(parts[5]), float(parts[6])  # Coordinates (x, y, z)
            ix, iy, iz = int(parts[7]), int(parts[8]), int(parts[9])  # Image flags (ix, iy, iz)
            # Append parsed atom data to the atoms list
            atoms.append((atom_id, mol_id, atom_type, (x, y, z), (ix, iy, iz)))
            
        # Parse bonds section
        if bond_section and line.strip() and not line.startswith("#"):
            parts = line.split()  # Split line into individual parts
            bond_id = int(parts[0])  # Bond ID
            bond_type = int(parts[1])  # Bond type
            atom1_id = int(parts[2])  # Atom ID of the first atom in the bond
            atom2_id = int(parts[3])  # Atom ID of the second atom in the bond
            # Append parsed bond data to the bonds list
            bonds.append((bond_id, bond_type, atom1_id, atom2_id))
        
        # Parse angles section
        if angle_section and line.strip() and not line.startswith("#"):
            parts = line.split()  # Split line into individual parts
            angle_id = int(parts[0])  # Angle ID
            angle_type = int(parts[1])  # Angle type
            atom1_id = int(parts[2])  # Atom ID of the first atom in the angle
            atom2_id = int(parts[3])  # Atom ID of the second atom in the angle
            atom3_id = int(parts[4])  # Atom ID of the third atom in the angle
            # Append parsed angle data to the angles list
            angles.append((angle_id, angle_type, atom1_id, atom2_id, atom3_id))
    
    # Return the parsed atoms, bonds, angles, and box bounds
    return atoms, bonds, angles, box_bounds

def Rg(atoms, bonds):
    # Create atom_id → coordinate mapping
    atom_coords = {atom[0]: np.array(atom[3]) for atom in atoms}

    # Build the bond graph
    G = nx.Graph()
    for _, _, atom1, atom2 in bonds:
        G.add_edge(atom1, atom2)

    # Analyze clusters
    clusters = list(nx.connected_components(G))
    rg_results = []

    for i, cluster in enumerate(clusters, start=1):
        if len(cluster) < 2:
            print(f"Cluster {i}: only 1 atom, skipping Rg calculation.")
            continue
        
        coords = np.array([atom_coords[atom_id] for atom_id in cluster if atom_id in atom_coords])
        
        if coords.size == 0:
            print(f"Cluster {i} has no coordinates.")
            continue

        com = np.mean(coords, axis=0)
        squared_distances = np.sum((coords - com) ** 2, axis=1)
        rg = np.sqrt(np.mean(squared_distances))
        
        rg_results.append(rg)
        print(f"Cluster {i}: N = {len(coords)}, Rg = {rg:.4f}")

    # Calculate and print average Rg
    if rg_results:
        avg_rg = np.mean(rg_results)
        print(f"\nAverage Rg across all clusters (N > 1): {avg_rg:.4f}")
    else:
        print("\nNo clusters with more than one atom.")


# Example usage: Read data from a LAMMPS restart file
# Replace "restart.data" with the path to your actual LAMMPS data file

file_path="restart.data"
atoms, bonds, angles, box_bounds = parse_lammps_data(file_path)
Rg(atoms,bonds)


