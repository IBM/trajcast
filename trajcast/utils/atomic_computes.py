from typing import Optional, Union

import ase.neighborlist
import numpy as np
import torch
from ase import Atoms


def cart2frac(cartesian_coords: np.array, lattice_vectors: np.array) -> np.array:
    inverse_lattice = np.linalg.inv(lattice_vectors)
    return np.dot(cartesian_coords, inverse_lattice)


def frac2cart(fractional_coords: np.array, lattice_vectors: np.array) -> np.array:
    return np.dot(fractional_coords, lattice_vectors)


def align_vectors_with_periodicity(
    vectors: np.array, lattice_vectors: np.array
) -> np.array:
    # transform the vector into fractional space
    fractional_vectors = cart2frac(vectors, lattice_vectors)

    # adjust the vector
    fractional_vectors[fractional_vectors > 0.5] -= 1.0
    fractional_vectors[fractional_vectors < -0.5] += 1.0

    # transform back and return
    return frac2cart(fractional_vectors, lattice_vectors)


def wrap_positions_back_to_box(pos: np.array, lattice_vectors: np.array) -> np.array:
    # transform vector into fractional coordinates
    fractional_coords = cart2frac(pos, lattice_vectors)

    # adjust the positions
    # adjust the vector
    fractional_coords[fractional_coords > 1.0] -= 1.0
    fractional_coords[fractional_coords < 0.0] += 1.0

    # transform back and return
    return frac2cart(fractional_coords, lattice_vectors)


def cart2frac_torch(
    cartesian_coords: torch.Tensor, lattice_vectors: torch.Tensor
) -> torch.Tensor:
    inverse_lattice = torch.inverse(lattice_vectors)
    return torch.matmul(cartesian_coords, inverse_lattice)


def frac2cart_torch(
    fractional_coords: torch.Tensor, lattice_vectors: torch.Tensor
) -> torch.Tensor:
    return torch.matmul(fractional_coords, lattice_vectors)


def cart2frac_torch_batch(
    cartesian_coords: torch.Tensor,
    lattice_vectors: torch.Tensor,
) -> torch.Tensor:
    inverse_lattice = torch.inverse(lattice_vectors)
    return torch.einsum("ni,nij->nj", cartesian_coords, inverse_lattice)


def frac2cart_torch_batch(
    fractional_coords: torch.Tensor,
    lattice_vectors: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("ni,nij->nj", fractional_coords, lattice_vectors)


def wrap_positions_back_to_box_torch(
    pos: torch.Tensor, lattice_vectors: torch.Tensor
) -> torch.Tensor:
    # transform vector into fractional coordinates
    fractional_coords = cart2frac_torch(pos, lattice_vectors)

    # adjust the positions
    # adjust the vector
    fractional_coords[fractional_coords > 1.0] -= 1.0
    fractional_coords[fractional_coords < 0.0] += 1.0

    # transform back and return
    return frac2cart_torch(fractional_coords, lattice_vectors)


def wrap_positions_back_to_box_torch_batch(
    pos: torch.Tensor, lattice_vectors: torch.Tensor, batch: torch.Tensor
) -> torch.Tensor:
    lattice_vectors = lattice_vectors.view(-1, 3, 3)[batch]
    # transform vector into fractional coordinates
    fractional_coords = cart2frac_torch_batch(pos, lattice_vectors)
    # adjust the positions
    # adjust the vector
    fractional_coords[fractional_coords > 1.0] -= 1.0
    fractional_coords[fractional_coords < 0.0] += 1.0

    # transform back and return
    return frac2cart_torch_batch(fractional_coords, lattice_vectors)


def cell_parameters_to_lattice_vectors(
    a: Optional[float] = None,
    b: Optional[float] = None,
    c: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    params: Optional[float] = None,
    tolerance: Optional[float] = 1e-6,
):
    if params is not None:
        if len(params) != 6:
            raise ValueError("params must be a 1D array-like with 6 elements.")
        a, b, c, alpha, beta, gamma = params

    # convert angles to from degrees to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    # Calculate the lattice vectors
    a_vector = np.array([a, 0, 0])
    b_vector = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0])
    c_x = c * np.cos(beta_rad)
    c_y = (b * c * np.cos(alpha_rad) - b_vector[0] * c_x) / b_vector[1]
    c_vector = np.array([c_x, c_y, np.sqrt(c**2 - c_x**2 - c_y**2)])

    # make sure now numeral issues are present:
    cell_vectors = np.stack([a_vector, b_vector, c_vector])
    cell_vectors[cell_vectors < tolerance] = 0
    return cell_vectors


def compute_kinetic_energy_for_individual_state(
    velocities: torch.Tensor,
    masses: torch.Tensor,
) -> torch.Tensor:
    # reshape if required
    if len(masses.size()) == 2:
        masses = masses.view(-1)
    dot_product = torch.sum(velocities * velocities, dim=1)
    return 0.5 * torch.sum(masses * dot_product)


def compute_com_velocity_for_individual_state(
    velocities: torch.Tensor,  # (N,3)
    masses: torch.Tensor,  # (N,)
    total_mass: torch.Tensor,  # (1)
) -> torch.Tensor:

    return torch.sum(masses * velocities, dim=0) / total_mass


def remove_angular_momentum_for_individual_state(
    positions: torch.Tensor,  # (N,3)
    velocities: torch.Tensor,  # (N,3)
    masses: torch.Tensor,  # (N,)
    total_mass: torch.Tensor,  # (1)
) -> torch.Tensor:

    # reshape if required
    if len(masses.size()) == 1:
        masses = masses.view(-1, 1)

    # compute momenta
    momenta = masses * velocities
    ang_mom, dist_com = compute_angular_momentum_for_individual_state(
        positions, momenta, masses, total_mass
    )

    # compute interia tensor
    inertia_tensor = compute_inertia_tensor_for_individual_state(
        masses=masses, dist_com=dist_com
    )

    angular_vel = torch.matmul(torch.inverse(inertia_tensor), (ang_mom))

    vel_adjust = torch.linalg.cross(angular_vel.unsqueeze(0), dist_com)

    return velocities - vel_adjust


def remove_linear_momentum_for_individual_state(
    velocities: torch.Tensor,  # (N,3)
    masses: torch.Tensor,  # (N,)
    total_mass: torch.Tensor,  # (1)
) -> torch.Tensor:

    # reshape masses if need be
    if len(masses.size()) == 1:
        masses = masses.view(-1, 1)

    com_velocity = compute_com_velocity_for_individual_state(
        velocities=velocities, masses=masses, total_mass=total_mass
    )

    return velocities - com_velocity


def compute_angular_momentum_for_individual_state(
    positions: torch.Tensor,  # (N,3)
    momenta: torch.Tensor,  # (N,3)
    masses: torch.Tensor,  # (N,1)
    total_mass: torch.Tensor,
):
    com = (masses * positions).sum(0) / total_mass
    dist_com = positions - com
    ang_mom = torch.linalg.cross(dist_com, momenta).sum(0)

    return ang_mom, dist_com


def compute_inertia_tensor_for_individual_state(
    masses: torch.Tensor,
    dist_com: Optional[torch.Tensor] = torch.tensor([]),
):
    mr = masses * dist_com
    mr2 = mr * dist_com
    Inert = torch.zeros(
        3, 3, dtype=torch.get_default_dtype(), device=masses.device
    )
    Inert[0, 0] = mr2[:, [1, 2]].sum()
    Inert[1, 1] = mr2[:, [0, 2]].sum()
    Inert[2, 2] = mr2[:, [0, 1]].sum()
    Inert[0, 1] = Inert[1, 0] = (-mr[:, 0] * dist_com[:, 1]).sum()
    Inert[0, 2] = Inert[2, 0] = (-mr[:, 0] * dist_com[:, 2]).sum()
    Inert[1, 2] = Inert[2, 1] = (-mr[:, 1] * dist_com[:, 2]).sum()

    return Inert


def compute_inertia_tensor_for_ase_atoms(ASEAtomsObject: Atoms) -> np.ndarray:
    """Get the total inertia tensor of the system.
    Adapted from ASE: https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.get_moments_of_inertia.

    Args:
        ASEAtomsObject (Atoms): Atoms for which we'd like to compute the inertia tensor.

    Returns:
        np.ndarray: Inertia tensor of the system.
    """
    com = ASEAtomsObject.get_center_of_mass()
    positions = ASEAtomsObject.get_positions()
    positions -= com  # translate center of mass to origin
    masses = ASEAtomsObject.get_masses()

    # Initialize elements of the inertial tensor
    I11 = I22 = I33 = I12 = I13 = I23 = 0.0
    for i in range(len(ASEAtomsObject)):
        x, y, z = positions[i]
        m = masses[i]

        I11 += m * (y**2 + z**2)
        I22 += m * (x**2 + z**2)
        I33 += m * (x**2 + y**2)
        I12 += -m * x * y
        I13 += -m * x * z
        I23 += -m * y * z

    return np.array([[I11, I12, I13], [I12, I22, I23], [I13, I23, I33]])


def compute_angular_momentum_for_ase_atoms(ASEAtomsObject: Atoms) -> np.ndarray:
    """Get the total angular momentum with respect to the center of mass.
    Adapted from ASE: https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.get_moments_of_inertia.

    Args:
        ASEAtomsObject (Atoms): Atoms for which we'd like to compute the angular momentum.

    Returns:
        np.ndarray: Angular momentum of the system.
    """

    com = ASEAtomsObject.get_center_of_mass()
    positions = ASEAtomsObject.get_positions()
    positions -= com

    return np.cross(
        positions,
        ASEAtomsObject.get_masses().reshape(-1, 1)
        * ASEAtomsObject.arrays["velocities"],
    ).sum(0)


def compute_com_velocity_for_ase_atoms(ASEAtomsObject: Atoms) -> np.ndarray:
    """Given an ase atoms this function calculates and returns the center of mass velocity.

    Args:
        ASEAtomsObject (Atoms): Atoms for which we'd like the center of mass velocity.

    Returns:
        np.ndarray: Center of mass velocity in cartesian coordinates.
    """

    return np.sum(
        ASEAtomsObject.get_masses().reshape(-1, 1)
        * ASEAtomsObject.arrays["velocities"],
        axis=0,
    ) / np.sum(ASEAtomsObject.get_masses())


def find_edges_for_periodic_system(
    node_positions: Union[torch.Tensor, np.ndarray],
    pbc: Union[torch.Tensor, np.ndarray],
    cell: Union[torch.Tensor, np.ndarray],
    r_cut: float = 5.0,
):
    """_summary_

    Args:
        node_positions (FIELD_TYPES[POSITIONS_KEY]): _description_
        pbc (FIELD_TYPES[PBC_KEY]): _description_
        cell (FIELD_TYPES[CELL_KEY]): _description_
        r_cut (float, optional): _description_. Defaults to 5.0.
    """

    # follow similar procedure as in NequiP
    temp_pos = (
        node_positions.numpy()
        if isinstance(node_positions, torch.Tensor)
        else node_positions
    )

    # use ASE to get neighbors
    edge_src, edge_dst, unit_shifts = ase.neighborlist.primitive_neighbor_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=temp_pos,
        cutoff=r_cut,
        self_interaction=False,  # in case cutoff extends to periodic image we'd like to have this interaction
    )

    # compute edge index in accordance with pytorch_geometric data
    edge_index = torch.vstack(
        [
            torch.from_numpy(edge_src).type(torch.long),
            torch.from_numpy(edge_dst).type(torch.long),
        ]
    )

    # from unit shifts to vectors

    shifts = torch.einsum(
        "ni,ij->nj",
        torch.from_numpy(unit_shifts).type(torch.get_default_dtype()),
        cell,
    )
    return edge_index, shifts
