import torch
from typing import Union, List
from trajcast.data._keys import ATOMIC_MASSES_KEY, VELOCITIES_KEY
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.model.forecast_tools import Temperature
from trajcast.utils.atomic_computes import compute_kinetic_energy_for_individual_state
from trajcast.utils.misc import GLOBAL_DEVICE


class CSVRThermostat(torch.nn.Module):
    """This thermostat adjusts the velocities every step to simulate the NVT ensemble
    while enforcing a canonical distribution. For details we refer to the original paper:
    Bussi, Donadio and Parrinello, J. Chem. Phys. 126, 014101(2007)
    """

    def __init__(
        self,
        target_temp: Union[float, int, List[float], List[int]],
        timestep: float,
        damping: float,
        temperature_handler: Temperature,
        n_steps: int,
    ):
        super().__init__()

        self.device = GLOBAL_DEVICE.device
        self.temp = temperature_handler
        self.n_dofs = self.temp._n_dofs

        # we follow here the implementation from LAMMPS starting with the definition of constant values:
        # exp const
        self.register_buffer(
            "c1",
            torch.exp(
                torch.tensor([-timestep / damping], dtype=torch.get_default_dtype())
            ),
        )
        if isinstance(target_temp, float) or isinstance(target_temp, int):
            self.register_buffer("start_temp", torch.tensor(float(target_temp)))
            self.temp_style = "CONST"

        else:
            if len(target_temp) != 2:
                raise TypeError(
                    "Temperature should be float or list with two floats (start and stop temperature)."
                )

            start_temp = float(target_temp[0])
            stop_temp = float(target_temp[1])
            self.temp_style = "RAMP" if start_temp != stop_temp else "CONST"
            self.register_buffer(
                "rate", torch.tensor((stop_temp - start_temp) / n_steps)
            )
            self.register_buffer("start_temp", torch.tensor(start_temp))

        # for sampling noise we initialise a gamma distribution
        if (self.n_dofs - 1) % 2 == 0:
            self.gamma_dist = torch.distributions.Gamma(
                concentration=(self.n_dofs - 1) / 2, rate=1.0
            )
        else:
            self.gamma_dist = torch.distributions.Gamma(
                concentration=(self.n_dofs - 2) / 2, rate=1.0
            )

    def forward(self, data: AtomicGraph, step: int) -> AtomicGraph:
        masses = data[ATOMIC_MASSES_KEY]
        velocities = data[VELOCITIES_KEY]

        # compute kinetic energy
        e_kin = (
            compute_kinetic_energy_for_individual_state(velocities, masses)
            * self.temp.conv_fac
        )
        # compute rescale_factor
        alpha = self._get_rescale_factor(e_kin, step)

        # rescale velocities accordingly
        data[VELOCITIES_KEY] *= alpha
        return data

    def _get_rescale_factor(
        self, e_kin_current: torch.Tensor, step: int
    ) -> torch.Tensor:

        if self.temp_style == "CONST":
            temp_target = self.start_temp

        else:
            temp_target = self.start_temp + self.rate * step

        # target kinetic energy, this is in eV
        e_kin_target = self.temp.to_kinetic_energy(temp_target)

        # compute constant c2 with c1 and kinetic energies
        c2 = (
            (torch.tensor(1.0) - self.c1) * e_kin_target / e_kin_current / self.n_dofs
        ).to(self.device)
        # draw random number from Gaussian distribution with unitary variance (R1)
        r1 = torch.randn(1, device=self.device)
        # draw n_dofs random numbers and sum the squares. As suggested in the original
        #  reference, this can be drawn directly from the gamma distribution.
        r2 = self._draw_sum_noises_from_gamma_dist(n_dofs_1=self.n_dofs - 1)

        alpha_2 = (
            self.c1
            + c2 * (r1 * r1 + r2)
            + torch.tensor(2.0) * r1 * (self.c1 * c2).sqrt()
        )

        return alpha_2.sqrt()

    def _draw_sum_noises_from_gamma_dist(self, n_dofs_1: int) -> torch.Tensor:
        if n_dofs_1 == 0:
            return torch.zeros(1, device=self.device)
        elif n_dofs_1 == 1:
            r2 = torch.randn(1, device=self.device)
            return r2 * r2
        elif n_dofs_1 % 2 == 0:
            return 2 * self.gamma_dist.sample().to(self.device)
        else:
            rr = torch.randn(1, device=self.device)
            return 2 * self.gamma_dist.sample().to(self.device) + rr * rr
