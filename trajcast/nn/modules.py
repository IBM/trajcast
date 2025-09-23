"""
Collection of individual ready-to-use modules to build a TrajCast model.
These can be either of message-passing character or employ a different approach.

Authors: Fabian Thiemann
"""

from typing import Dict, Optional, Union

import torch
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import Irrep, Irreps
from torch_geometric.data import Batch
from torch_scatter import scatter

from trajcast.data._keys import (
    ATOMIC_MASSES_KEY,
    NODE_FEATURES_KEY,
    TIMESTEP_ENCODING_KEY,
    TOTAL_MASS_KEY,
    VELOCITIES_KEY,
)
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.nn._graph_module_irreps import GraphModuleIrreps
from trajcast.utils.atomic_computes import (
    compute_angular_momentum_for_individual_state,
    compute_inertia_tensor_for_individual_state,
)
from trajcast.utils.misc import (
    ACTIVATION_FUNCTIONS,
    mlp_config_from_dictionary,
)


class ConservationLayer(GraphModuleIrreps, torch.nn.Module):
    """This layer is used to make sure the model preserves (like NVE simulations) the total energy and the total linear.
    It should be called after the last layer to update the velocities to remove the total linear momentum accumulated between input and output.

    In addition to the total linear momentum, the angular momentum is (at least for isolated molecules) conserved as well.
    Therefore, using the corresponding keyword, we allow also conserve the angular momentum of the system.
    """

    kin_to_pot_unit_conv = {
        "metal": 0.00010364269574711573,
        "real": 2390.057361376673040153,
    }

    def __init__(
        self,
        input_field: Optional[str] = "target",
        velocity_field: Optional[str] = VELOCITIES_KEY,
        index_disp_target: Optional[int] = 0,
        index_vel_target: Optional[int] = 3,
        irreps_in: Optional[Dict[str, Irreps]] = {},
        conserve_angular: Optional[bool] = False,
        units: Optional[bool] = "real",
        vel_norm_const: Optional[Union[float, torch.Tensor]] = 1.0,
        disp_norm_const: Optional[Union[float, torch.Tensor]] = 1.0,
        net_lin_mom: Optional[torch.Tensor] = torch.tensor([]),
        net_ang_mom: Optional[torch.Tensor] = torch.tensor([]),
    ):
        """Initialisation of the layer.

        Args:
            input_field (Optional[str], optional): Name of the field in the graph were the raw predictions lie. Defaults to "target".
            velocity_field (Optional[str], optional): Name of the field were the velocities of the input are. Defaults to VELOCITIES_KEY.
            index_vel_target (Optional[int], optional): Index where the velocities start in the input field. Defaults to 3.
                Note: Usually, the target comprises both displacements and new velocities, that's why we need to tell the model which part
                of the target we'd like to modify (only the velocities).
            irreps_in (Optional[Dict[str, Irreps]], optional): Dictionary of all irreps. Defaults to {}.
            conserve_angular (Optiona[int], optional): Whether to conserve the angular momentum too.
                Note: This should only be true if you work on isolated molecules for now.
            vel_norm_const (Optional[float], optional): By which factor to scale the input velocities. Defaults to 1.0.
                Note: Our model predicts normalised velocities. If the input velocities were not normalised we need to
                normalise them by the same factor as the 'update_velocities' used in training.
            disp_norm_const (Optional[float], optional): The factor the displacements are normalised by. Defaults to 1.0.
                Note: Our model predicts normalised displacements. To obtain the new center of mass which is important for
                computing the total angular momentum they need to be unnormalised and the positions updated.
            net_lin_mom (Optional[torch.Tensor], optional): The target net linear momentum. If set, the velocities will be adjusted to
                match this net linear momentum.
            net_ang_mom (Optional[torch.Tensor], optional): The target net angular momentum. If set, the velocities will be adjusted to
                match this net angular momentum.
        """
        super().__init__()
        self.input_field = input_field
        self.velocity_field = velocity_field
        self.index_vel = index_vel_target
        self.index_disp = index_disp_target
        self.angular = conserve_angular

        if isinstance(vel_norm_const, float):
            vel_norm_const = torch.tensor(vel_norm_const)

        if isinstance(disp_norm_const, float):
            disp_norm_const = torch.tensor(disp_norm_const)

        self.register_buffer(
            "prefactor_vel",
            vel_norm_const,
        )

        self.register_buffer(
            "prefactor_disps",
            disp_norm_const,
        )

        self.register_buffer(
            "net_lin_mom",
            net_lin_mom,
        )

        self.register_buffer(
            "net_ang_mom",
            net_ang_mom,
        )

        self._init_irreps(irreps_in)

    def forward(self, data: AtomicGraph) -> AtomicGraph:
        # compute net linear momentum of input data: can be done later in the preprocessing of the data
        data = self._preserve_linear_momentum(data)

        if self.angular:
            data = self._preserve_angular_momentum(data)
        data[self.input_field][
            :, self.index_vel : self.index_vel + 3
        ] /= self.prefactor_vel

        return data

    def _preserve_angular_momentum(self, data: AtomicGraph) -> AtomicGraph:
        """Preserves the angular momentum by modifying the predicted velocities.
        Note: Only works for isolated molecules at this stage!
        Args:
            data (AtomicGraph): _description_

        Returns:
            AtomicGraph: _description_
        """

        momenta_out = (
            data[ATOMIC_MASSES_KEY]
            * data[self.input_field][:, self.index_vel : self.index_vel + 3]
        )

        # get new positions based on predicted update
        new_pos = (
            data.pos
            + data[self.input_field][:, self.index_disp : self.index_disp + 3]
            * self.prefactor_disps
        )

        masses = data[ATOMIC_MASSES_KEY]
        total_mass = data[TOTAL_MASS_KEY]

        if isinstance(data, Batch):

            def compute_angular_momentum_for_batch_state(
                positions: torch.Tensor, momenta: torch.Tensor
            ):
                # compute center of mass
                com = scatter(
                    masses * positions,
                    index=data.batch,
                    dim=0,
                    reduce="sum",
                ) / total_mass.view(-1, 1)
                # compute distance from center of mass. Note, this does not account for pbcs.
                dist_com = positions - com[data.batch]

                # compute angular momentum
                ang_mom = scatter(
                    torch.linalg.cross(dist_com, momenta),
                    index=data.batch,
                    dim=0,
                    reduce="sum",
                )

                return ang_mom, dist_com

            if self.net_ang_mom.numel() > 0:
                batch_size = total_mass.size(0)
                ang_mom_in = self.net_ang_mom.repeat(batch_size, 1)
            else:
                momenta_in = masses * data[self.velocity_field]
                ang_mom_in, _ = compute_angular_momentum_for_batch_state(
                    data.pos, momenta_in
                )
            ang_mom_out, dist_com_out = compute_angular_momentum_for_batch_state(
                new_pos, momenta_out
            )

            # compute inertia tensor
            mr = masses * dist_com_out
            mr2 = mr * dist_com_out
            Inert = torch.zeros(data.ptr.size(0) - 1, 3, 3, device=masses.device)
            Inert[:, 0, 0] = scatter(
                mr2[:, [1, 2]].sum(1), index=data.batch, dim=0, reduce="sum"
            )
            Inert[:, 1, 1] = scatter(
                mr2[:, [0, 2]].sum(1), index=data.batch, dim=0, reduce="sum"
            )
            Inert[:, 2, 2] = scatter(
                mr2[:, [0, 1]].sum(1), index=data.batch, dim=0, reduce="sum"
            )
            Inert[:, 0, 1] = Inert[:, 1, 0] = scatter(
                -mr[:, 0] * dist_com_out[:, 1], index=data.batch, dim=0, reduce="sum"
            )
            Inert[:, 0, 2] = Inert[:, 2, 0] = scatter(
                -mr[:, 0] * dist_com_out[:, 2], index=data.batch, dim=0, reduce="sum"
            )
            Inert[:, 1, 2] = Inert[:, 2, 1] = scatter(
                -mr[:, 1] * dist_com_out[:, 2], index=data.batch, dim=0, reduce="sum"
            )

            angular_vel_diff = torch.einsum(
                "ijk,ik->ij", torch.inverse(Inert), ang_mom_in - ang_mom_out
            )

            vel_adjust = torch.linalg.cross(angular_vel_diff[data.batch], dist_com_out)

        else:
            if self.net_ang_mom.numel() > 0:
                ang_mom_in = self.net_ang_mom
            else:
                momenta_in = masses * data[self.velocity_field]
                ang_mom_in, _ = compute_angular_momentum_for_individual_state(
                    data.pos, momenta_in, masses, total_mass
                )

            ang_mom_out, dist_com_out = compute_angular_momentum_for_individual_state(
                new_pos, momenta_out, masses, total_mass
            )
            # Inertia tensor
            Inert = compute_inertia_tensor_for_individual_state(masses, dist_com_out)

            # difference
            angular_vel_diff = torch.matmul(
                torch.inverse(Inert), (ang_mom_in - ang_mom_out)
            )
            vel_adjust = torch.linalg.cross(angular_vel_diff.unsqueeze(0), dist_com_out)

        data[self.input_field][:, self.index_vel : self.index_vel + 3] += vel_adjust

        return data

    def _preserve_linear_momentum(
        self,
        data: AtomicGraph,
    ) -> AtomicGraph:
        vel_out = (
            data[self.input_field][:, self.index_vel : self.index_vel + 3]
            * self.prefactor_vel
        )
        momenta_out = data[ATOMIC_MASSES_KEY] * vel_out
        if isinstance(data, Batch):
            if self.net_lin_mom.numel() > 0:
                batch_size = data[TOTAL_MASS_KEY].size(0)
                net_lin_mom_in = self.net_lin_mom.repeat(batch_size, 1)
            else:
                momenta_in = data[ATOMIC_MASSES_KEY] * data[self.velocity_field]
                net_lin_mom_in = scatter(
                    momenta_in, index=data.batch, dim=0, reduce="sum"
                )

            net_lin_mom_out = scatter(
                momenta_out, index=data.batch, dim=0, reduce="sum"
            )
            vel_adjust = (net_lin_mom_in - net_lin_mom_out) / data[TOTAL_MASS_KEY].view(
                -1, 1
            )
            vel_adjust = vel_adjust[data.batch]

            if torch.all(net_lin_mom_in == 0.0):
                disps = data[self.input_field][:, self.index_disp : self.index_disp + 3]
                disp_by_mass = data[ATOMIC_MASSES_KEY] * disps
                disp_com = scatter(disp_by_mass, index=data.batch, dim=0, reduce="sum")
                disp_adjust = (disp_com) / data[TOTAL_MASS_KEY].view(-1, 1)
                disp_adjust = disp_adjust[data.batch]
                data[self.input_field][:, self.index_disp : self.index_disp + 3] = (
                    disps - disp_adjust
                )
        else:
            net_lin_mom_in = self.net_lin_mom
            if net_lin_mom_in.numel() == 0:
                momenta_in = data[ATOMIC_MASSES_KEY] * data[self.velocity_field]
                net_lin_mom_in = momenta_in.sum(0)
            net_lin_mom_out = momenta_out.sum(0)
            vel_adjust = (net_lin_mom_in - net_lin_mom_out) / data[TOTAL_MASS_KEY]

            if torch.all(net_lin_mom_in == 0.0):
                disps = data[self.input_field][:, self.index_disp : self.index_disp + 3]
                disp_by_mass = data[ATOMIC_MASSES_KEY] * disps
                disp_adjust = disp_by_mass.sum(0) / data[TOTAL_MASS_KEY]
                data[self.input_field][:, self.index_disp : self.index_disp + 3] = (
                    disps - disp_adjust
                )

        data[self.input_field][:, self.index_vel : self.index_vel + 3] = (
            vel_out + vel_adjust
        )

        return data

    def _compute_kinetic_energy_for_batch_state(
        self,
        velocities: torch.Tensor,
        masses: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        dot_product = torch.sum(velocities * velocities, dim=1)
        return 0.5 * scatter(masses * dot_product, index=batch, reduce="sum")


class ForecastHorizonConditioning(GraphModuleIrreps, torch.nn.Module):
    """This layer updates the embeddings obtained in the message passing mechanism based on forecast horizon."""

    def __init__(
        self,
        node_features_field: Optional[str] = NODE_FEATURES_KEY,
        timestep_embedding_field: Optional[str] = TIMESTEP_ENCODING_KEY,
        output_field: Optional[str] = NODE_FEATURES_KEY,
        irreps_in: Optional[Dict[str, Irreps]] = {},
        fc_kwargs: Optional[Dict] = {},
        activation_function_gate: Optional[str] = "identity",
    ):
        super().__init__()

        self.node_features_field = node_features_field
        self.timestep_embedding_field = timestep_embedding_field
        self.output_field = output_field

        n_hidden_channels = irreps_in[self.node_features_field].count("0e")

        if n_hidden_channels != irreps_in[self.node_features_field].count("1o"):
            raise TypeError(
                "Not the same scalars as vectors in {self.node_features_field}."
            )

        irreps_gates = Irreps(f"{n_hidden_channels}x0e")
        irreps_vectors = Irreps(f"{n_hidden_channels}x1o")
        act_gates = [ACTIVATION_FUNCTIONS[activation_function_gate]]

        # extract information from kwargs
        neurons_per_layer, act_mlp = (
            mlp_config_from_dictionary(fc_kwargs)
            if fc_kwargs
            else ([16], torch.nn.SiLU())
        )

        # add potentially the raise of an error if more than scalars
        self.fc = FullyConnectedNet(
            [n_hidden_channels * 2] + neurons_per_layer + [n_hidden_channels],
            act_mlp,
        )

        self.gate = Gate(
            "",
            [],
            irreps_gates,
            act_gates,
            irreps_vectors,
        )

        irreps_out = {self.node_features_field: f"{n_hidden_channels}x1o"}
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

        # get the indices scalars and vectors in node features
        for j, (_, ir) in enumerate(self.irreps_in[self.node_features_field]):
            if ir == Irrep(0, 1):
                self.scalar_slice = self.irreps_in[self.node_features_field].slices()[j]
            elif ir == Irrep(1, -1):
                self.vector_slice = self.irreps_in[self.node_features_field].slices()[j]

    def forward(self, data: AtomicGraph) -> AtomicGraph:
        if isinstance(data, Batch):
            # concat timestep encoding to scalars
            updated_scalars = torch.cat(
                [
                    data[self.node_features_field][:, self.scalar_slice],
                    data[self.timestep_embedding_field][data.batch],
                ],
                dim=1,
            )

        else:
            updated_scalars = torch.cat(
                [
                    data[self.node_features_field][:, self.scalar_slice],
                    data[self.timestep_embedding_field].expand(data.num_nodes, -1),
                ],
                dim=1,
            )

        # pass updated scalars through mlp
        updated_scalars = self.fc(updated_scalars)
        vectors = data[self.node_features_field][:, self.vector_slice]
        input_gate = torch.cat([updated_scalars, vectors], dim=1)
        data[self.output_field] = self.gate(input_gate)

        return data
