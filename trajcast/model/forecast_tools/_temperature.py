import torch
from ase.units import J, kB

from trajcast.data._keys import ATOMIC_MASSES_KEY, VELOCITIES_KEY
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.model.forecast_tools._units import UNITS
from trajcast.utils.atomic_computes import compute_kinetic_energy_for_individual_state
from trajcast.utils.misc import convert_units


class Temperature(torch.nn.Module):

    def __init__(self, units: str, n_atoms: int, n_extra_dofs: int) -> None:
        super().__init__()

        self._n_extra_dofs = n_extra_dofs
        self._n_dofs = 3 * n_atoms - n_extra_dofs

        # We need to convert kinetic energy into temperature
        # to this end we set up a conversion factor to handle
        # th unit conversion from kinetic energy to temperature
        # check with units
        if units not in list(UNITS.keys()):
            raise KeyError(f"Unit {units} not allowed.")
        units = UNITS[units]
        unit_velocity = units.get("velocity")
        unit_mass = units.get("mass")
        units_distance_time = unit_velocity.replace(" ", "").split("/")

        self.register_buffer(
            "conv_fac",
            torch.tensor(
                [
                    (
                        convert_units(origin=units_distance_time[0], target="meter")
                        / convert_units(origin=units_distance_time[1], target="s")
                    )
                    ** 2
                    * convert_units(origin=unit_mass, target="kg")
                    * J
                ],
                dtype=torch.get_default_dtype(),
            ),
        )

        self.register_buffer("kB", torch.tensor(kB))

    def to_kinetic_energy(self, temperature: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [0.5 * self._n_dofs * temperature * self.kB],
            dtype=torch.get_default_dtype(),
            device=temperature.device,
        )

    def from_kinetic_energy(self, kinetic_energy: torch.Tensor) -> float:
        t = 2 * kinetic_energy * self.conv_fac / (self._n_dofs * self.kB)
        return t.item()

    def from_velocities_masses(
        self, velocities: torch.Tensor, masses: torch.Tensor
    ) -> float:
        kinetic_energy = compute_kinetic_energy_for_individual_state(velocities, masses)
        return self.from_kinetic_energy(kinetic_energy)

    def forward(self, data: AtomicGraph) -> float:
        masses = data[ATOMIC_MASSES_KEY]
        velocities = data[VELOCITIES_KEY]

        return self.from_velocities_masses(velocities, masses)
