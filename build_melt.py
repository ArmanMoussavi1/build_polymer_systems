import numpy as np


class PolymerChain:
    def __init__(self, chain_id, num_atoms, bond_length, start_position, existing_positions):
        self.chain_id = chain_id
        self.num_atoms = num_atoms
        self.bond_length = bond_length
        self.positions = np.zeros((num_atoms, 3))
        self.atom_types = np.zeros(num_atoms, dtype=int)
        self.atom_ids = np.arange(chain_id * num_atoms, chain_id * num_atoms + num_atoms)
        self.positions[0] = start_position
        self.atom_types[:] = 1  # Set all atoms to type 1
        self._generate_positions(existing_positions)

    def _generate_positions(self, existing_positions):
        for i in range(1, self.num_atoms):
            while True:
                move = self._generate_random_direction() * self.bond_length
                new_position = self.positions[i - 1] + move
                if not self._is_overlap(new_position, existing_positions):
                    self.positions[i] = new_position
                    existing_positions.add(tuple(new_position))
                    break

    @staticmethod
    def _generate_random_direction():
        direction = np.random.uniform(-1, 1, 3)
        norm = np.linalg.norm(direction)
        return direction / norm if norm > 0 else PolymerChain._generate_random_direction()

    @staticmethod
    def _is_overlap(position, existing_positions):
        return tuple(position) in existing_positions


class SimulationBox:
    def __init__(self, box_size, num_chains, num_atoms, bond_length):
        self.box_size = box_size
        self.num_chains = num_chains
        self.num_atoms = num_atoms
        self.bond_length = bond_length
        self.chains = []
        self.existing_positions = set()

        self._initialize_chains()

    def _initialize_chains(self):
        for chain_id in range(self.num_chains):
            # Random start position within the box, ensuring it doesn't overlap with other chains
            start_position = np.random.uniform(0, self.box_size, 3)
            chain = PolymerChain(chain_id, self.num_atoms, self.bond_length, start_position, self.existing_positions)
            self.chains.append(chain)

    def generate_bonds(self):
        bonds = []
        bond_id = 1
        for chain in self.chains:
            for i in range(len(chain.atom_ids) - 1):
                bonds.append((bond_id, 1, chain.atom_ids[i], chain.atom_ids[i + 1]))
                bond_id += 1
        return bonds

    def generate_angles(self):
        angles = []
        angle_id = 1
        for chain in self.chains:
            for i in range(1, len(chain.atom_ids) - 1):
                angles.append((angle_id, 1, chain.atom_ids[i - 1], chain.atom_ids[i], chain.atom_ids[i + 1]))
                angle_id += 1
        return angles

    def calculate_box_dimensions(self):
        all_positions = np.concatenate([chain.positions for chain in self.chains])
        min_coords = np.min(all_positions, axis=0)
        max_coords = np.max(all_positions, axis=0)
        # Set buffer to ensure the box is large enough to contain all chains
        buffer = 5  # Fixed buffer value
        return min_coords[0] - buffer, max_coords[0] + buffer, \
               min_coords[1] - buffer, max_coords[1] + buffer, \
               min_coords[2] - buffer, max_coords[2] + buffer

    def write_lammps_input(self, filename):
        bonds = self.generate_bonds()
        angles = self.generate_angles()
        xlo, xhi, ylo, yhi, zlo, zhi = self.calculate_box_dimensions()

        with open(filename, 'w') as file:
            num_atoms = self.num_atoms * self.num_chains
            num_bonds = len(bonds)
            num_angles = len(angles)
            num_atom_types = 1  # Only 1 type of atom
            num_bond_types = 1
            num_angle_types = 1

            file.write(f"# Hydrogel Model with Self-Avoiding Random Walk\n\n")
            file.write(f"{num_atoms} atoms\n{num_bonds} bonds\n{num_angles} angles\n")
            file.write("0 dihedrals\n0 impropers\n\n")
            file.write(f"{num_atom_types} atom types\n{num_bond_types} bond types\n{num_angle_types} angle types\n")
            file.write(f"{xlo} {xhi} xlo xhi\n{ylo} {yhi} ylo yhi\n{zlo} {zhi} zlo zhi\n\n")

            file.write("Atoms\n\n")
            for chain in self.chains:
                for atom_id, atom_type, position in zip(chain.atom_ids, chain.atom_types, chain.positions):
                    file.write(f"{atom_id} {chain.chain_id + 1} {atom_type} 0.0 {position[0]} {position[1]} {position[2]}\n")

            file.write("\nBonds\n\n")
            for bond in bonds:
                file.write(f"{bond[0]} {bond[1]} {bond[2]} {bond[3]}\n")

            file.write("\nAngles\n\n")
            for angle in angles:
                file.write(f"{angle[0]} {angle[1]} {angle[2]} {angle[3]} {angle[4]}\n")

            file.write("\nMasses\n\n")
            file.write("1 1.0\n")


def main(N, num_chains):
    box_size = np.array([50.0, 50.0, 50.0])
    bond_length = 1.5

    sim_box = SimulationBox(box_size, num_chains, N, bond_length)
    sim_box.write_lammps_input(f'N{N}_chains{num_chains}.data')
    print("LAMMPS output complete.")


# Run the simulation
N = 400
num_chains = 100
main(N, num_chains)
