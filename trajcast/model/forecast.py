import os
import warnings
from typing import Dict, Optional
from tqdm import tqdm
import ase.io
import torch
import yaml
from ase import Atoms

from trajcast.data._keys import (
    ARCHITECTURE_KEY,
    CELL_KEY,
    CONFIG_KEY,
    EXTRA_DOF_KEY,
    FILENAME_KEY,
    FRAME_KEY,
    MODEL_KEY,
    MODEL_TYPE_KEY,
    RUN_KEY,
    SET_MOMENTA_KEY,
    TEMPERATURE_KEY,
    THERMOSTAT_KEY,
    TIMESTEP_KEY,
    TYPE_MAPPER_KEY,
    UNITS_KEY,
    VELOCITIES_KEY,
    WRITE_TRAJECTORY_KEY,
    ZERO_MOMENTUM_KEY,
)
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.data.wrappers._lammps import lammps_dump_to_ase_atoms
from trajcast.model.forecast_tools import (
    CSVRThermostat,
    Temperature,
    ZeroMomentum,
    init_velocity,
)
from trajcast.model.models import EfficientTrajCastModel, FlexibleModel, TrajCastModel
from trajcast.nn.modules import ConservationLayer
from trajcast.utils.atomic_computes import wrap_positions_back_to_box_torch
from trajcast.utils.misc import (
    GLOBAL_DEVICE,
    convert_ase_atoms_to_dictionary,
    format_values_in_dictionary,
)


class Forecast:
    """This our engine for running TrajCast models and generating trajectories. Similar to classic
    MD engines like LAMMPS or GROMACS we only need to specify a few things. These include: the initial configuration,
    the timestep, the temperature, the length of the simulation, the model we are using etc.

    Raises:
        KeyError: _description_
        AttributeError: _description_
        FileNotFoundError: _description_
        TypeError: _description_
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """

    _allowed_attrs = [
        TEMPERATURE_KEY,
        CELL_KEY,
        MODEL_KEY,
        MODEL_TYPE_KEY,
        ARCHITECTURE_KEY,
        TIMESTEP_KEY,
        TYPE_MAPPER_KEY,
        RUN_KEY,
        UNITS_KEY,
        CONFIG_KEY,
        ZERO_MOMENTUM_KEY,
        WRITE_TRAJECTORY_KEY,
        THERMOSTAT_KEY,
        SET_MOMENTA_KEY,
        VELOCITIES_KEY,
        EXTRA_DOF_KEY,
    ]

    _model_types = {
        "TrajCast": TrajCastModel,
        "Flexible": FlexibleModel,
        "EfficientTrajCast": EfficientTrajCastModel,
    }

    def __init__(self, protocol: Dict):
        GLOBAL_DEVICE.device = protocol.pop("device", "cpu")
        self.device = GLOBAL_DEVICE.device
        torch.manual_seed(protocol.pop("seed", 1705))
        self.protocol = protocol
        # from the protocol read all attributes
        for key, value in self.protocol.items():
            if key not in self._allowed_attrs:
                raise KeyError(f"Key '{key}' is not allowed.")

        # check that all necessary fields are given
        # list of all fields not given
        missing_fields = [
            attr
            for attr in self._allowed_attrs
            if not self.protocol.get(attr)
            and attr
            not in [
                CELL_KEY,
                ARCHITECTURE_KEY,
                MODEL_KEY,
                MODEL_TYPE_KEY,
                TEMPERATURE_KEY,
                WRITE_TRAJECTORY_KEY,
                ZERO_MOMENTUM_KEY,
                SET_MOMENTA_KEY,
                THERMOSTAT_KEY,
                VELOCITIES_KEY,
                EXTRA_DOF_KEY,
                TYPE_MAPPER_KEY,
            ]
        ]

        if missing_fields:
            raise AttributeError(
                "\n".join(
                    [
                        f"Error: {field} not found, please specify"
                        for field in missing_fields
                    ]
                )
            )

        if not self.protocol.get(MODEL_KEY) and not self.protocol.get(ARCHITECTURE_KEY):
            raise AttributeError(
                "You must specify either the model or at least the architecture."
            )

        # now we check whether the configuration and the model are paths to file or ASE Atoms and torch objects
        # first the configuration
        atomic_config = self.protocol.get(CONFIG_KEY)
        if isinstance(atomic_config, AtomicGraph):
            self.start_graph = atomic_config

        elif isinstance(atomic_config, Atoms):
            ase_atoms = atomic_config

        elif isinstance(atomic_config, str):
            # check whether path exists:
            if not os.path.exists(atomic_config):
                raise FileNotFoundError("File with atomic coordinates not found.")

            # print a warning here
            warnings.warn(
                f"Passing the {CONFIG_KEY} without any arguments is deprecated as ase might read it"
                "in wrongly if not given as extxyz. We recommend passing arguments in dictionary format",
                DeprecationWarning,
            )

            # otherwise let's try to read it in via ASE
            ase_atoms = ase.io.read(atomic_config)

        elif isinstance(atomic_config, Dict):
            # in case we have a dictionary the read in is a bit more complex
            dic = atomic_config
            # let us start by getting the filename
            filename = dic.get(FILENAME_KEY)
            if not os.path.exists(filename):
                raise FileNotFoundError("File with atomic coordinates not found.")

            # get the wrapper key word
            wrapper = dic.get("wrapper")
            wrapper_kwargs = dic.get("wrapper_kwargs")
            index = dic.get("index", "0")
            # read in ase atoms (similar to ASETrajectory)
            if wrapper:
                ase_atoms = {"lammps": lammps_dump_to_ase_atoms}[wrapper](
                    path_to_file=filename,
                    index=index,
                    **wrapper_kwargs,
                )
            else:
                warnings.warn(
                    "No wrapper given, will just use standard ase.",
                    UserWarning,
                )

                # otherwise let's try to read it in via ASE
                ase_atoms = ase.io.read(filename, index=index)

        else:
            raise TypeError(
                f"{CONFIG_KEY} needs to be the path to the starting configuration or an ASE.Atoms object."
            )

        # do the same with the model
        model_given = self.protocol.get(MODEL_KEY)
        architecture_given = self.protocol.get(ARCHITECTURE_KEY)
        model_type = self.protocol.get(MODEL_TYPE_KEY)

        if model_given:
            if isinstance(model_given, torch.nn.Module):
                self.predictor = model_given

            elif isinstance(model_given, str) and architecture_given:
                # check whether path exists:
                if not os.path.exists(model_given):
                    raise FileNotFoundError("File with model parameters not found.")

                if not model_type:
                    raise KeyError(
                        "If you pass model parameters we also need to know the model type."
                    )

                # instantiate
                model = self._model_types[model_type]

                # we need the architecture as well
                if isinstance(architecture_given, str):
                    if not os.path.exists(architecture_given):
                        raise FileNotFoundError(
                            "You need to specify the model architecture as well."
                        )
                    # initialise model with random parameters...
                    self.predictor = model.build_from_yaml(filename=architecture_given)

                # architecture could also be a dictionary
                elif isinstance(architecture_given, Dict):
                    if not model_type:
                        raise KeyError("We also need to know the model type.")
                    model = self._model_types[model_type]

                    self.predictor = model(config=architecture_given)

                else:
                    raise TypeError(
                        "Architecture needs to be provided either from file or dictionary."
                    )

                # ... now load the parameters
                self.predictor.load_state_dict(torch.load(model_given))
            else:
                raise AttributeError(
                    "Architecture needs to be provided either from file or dictionary."
                )

        elif not model_given and isinstance(architecture_given, str):
            if not os.path.exists(architecture_given):
                raise FileNotFoundError(
                    "You need to specify the correct path to the architecture."
                )

            if not model_type:
                raise KeyError("We also need to know the model type.")
            model = self._model_types[model_type]

            self.predictor = model.build_from_yaml(filename=architecture_given)

        # get cutoff from model
        self.cutoff = self.predictor.edge_cutoff.item()

        # now get the ASE object/AtomicGraph
        if not hasattr(self, "start_graph"):
            # generate graph, we need a cutoff for this
            if not self.protocol.get(TYPE_MAPPER_KEY):
                warnings.warn(
                    "Note that no mapping for atom types to types within the model is specified, will do this automatically."
                    f"We recommend passing arguments in dictionary format argument {TYPE_MAPPER_KEY}",
                    UserWarning,
                )

            self.start_graph = AtomicGraph.from_atoms_dict(
                format_values_in_dictionary(convert_ase_atoms_to_dictionary(ase_atoms)),
                r_cut=self.cutoff,
                atom_type_mapper=self.protocol.get(TYPE_MAPPER_KEY),
            )

        self.predictor.to(self.device)
        # set model into evaluation model
        self.predictor.eval()

        # once the predictor is loaded we need to check if renormalization is required
        # instantiate renormalization with default
        self.disp_scale = 1.0
        self.vel_scale = 1.0
        if hasattr(self.predictor, "rms_targets"):
            self.disp_scale = self.predictor.rms_targets[0]
            self.vel_scale = self.predictor.rms_targets[1]

        # at the end update the timestep we would like to run
        timestep = self.protocol.get(TIMESTEP_KEY)
        self.start_graph[TIMESTEP_KEY] = torch.tensor([timestep])
        # init a batch tensor for torch_nl neighborlist
        self.start_graph["batch"] = torch.zeros(
            self.start_graph.num_nodes, dtype=torch.long
        )
        # put graph on device
        self.start_graph.to(self.device)

        # here we check whether net momenta are set and pass this to the model
        set_momenta_dict = self.protocol.get(SET_MOMENTA_KEY)
        if set_momenta_dict:
            # get conservation layer
            if hasattr(self.predictor, "layers"):

                if not [
                    name
                    for name, module in dict(
                        self.predictor.layers.named_children()
                    ).items()
                    if isinstance(module, ConservationLayer)
                ]:
                    raise AttributeError(
                        "You cannot set the net momenta without having a ConservationLayer to control them."
                    )

                self.predictor.layers.Conservation.net_lin_mom = set_momenta_dict.get(
                    "linear"
                )
                self.predictor.layers.Conservation.net_ang_mom = set_momenta_dict.get(
                    "angular"
                )

            else:
                if not hasattr(self.predictor, "_conservation"):
                    raise AttributeError(
                        "You cannot set the net momenta without having a ConservationLayer to control them."
                    )

                self.predictor._conservation.net_lin_mom = set_momenta_dict.get(
                    "linear"
                )
                self.predictor._conservation.net_ang_mom = set_momenta_dict.get(
                    "angular"
                )

        # here all additional simulation settings are handled by the class and default values are given
        # linear momentum removal
        self.momentum = None
        if self.protocol.get(ZERO_MOMENTUM_KEY):
            self.momentum = ZeroMomentum(
                settings=self.protocol.get(ZERO_MOMENTUM_KEY)
            ).to(self.device)

        # initialise temperature
        # compute constrained degrees of freedom
        extra_dofs = self.protocol.get(EXTRA_DOF_KEY)
        if extra_dofs:
            if not isinstance(extra_dofs, int):
                raise KeyError(
                    f"Degrees of freedom to be substracted (keyword: {EXTRA_DOF_KEY}) must be an integer."
                )
            n_dof = extra_dofs
        else:
            n_dof = 0
            if hasattr(self.predictor, "layers"):
                # check if there is momentum conservation
                if isinstance(self.predictor.layers[-1], ConservationLayer) or (
                    self.momentum and self.momentum.zero_linear
                ):
                    n_dof += 3

                if (
                    hasattr(self.predictor.layers, "Conservation")
                    and self.predictor.layers.Conservation.net_ang_mom.numel() > 0
                ) or (self.momentum and self.momentum.zero_angular):
                    n_dof += 3

            else:
                # check if there is momentum conservation
                if hasattr(self.predictor, "_conservation") or (
                    self.momentum and self.momentum.zero_linear
                ):
                    n_dof += 3

                if (self.momentum and self.momentum.zero_angular) or (
                    hasattr(self.predictor, "_conservation")
                    and self.predictor._conservation.net_ang_mom.numel() > 0
                ):
                    n_dof += 3

        self.temp = Temperature(
            units=self.protocol.get(UNITS_KEY),
            n_atoms=self.start_graph.num_nodes,
            n_extra_dofs=n_dof,
        ).to(self.device)

        # check regarding thermostat
        thermostat = self.protocol.get(THERMOSTAT_KEY)
        if thermostat:
            self.nvt = True
            if isinstance(thermostat, Dict):
                tau = thermostat.get("Tdamp", timestep * 100)
            elif isinstance(thermostat, bool):
                tau = timestep * 100

            else:
                raise TypeError(
                    f"Either dictionary or boolean should be passed to {THERMOSTAT_KEY}."
                )

            self.thermo = CSVRThermostat(
                target_temp=self.protocol.get(TEMPERATURE_KEY),
                timestep=timestep,
                damping=tau,
                temperature_handler=self.temp,
                n_steps=self.protocol.get(RUN_KEY),
            ).to(self.device)

        else:
            self.nvt = False

        # check whether velocities should be initialised from distribution and if so how
        init_vel = self.protocol.get(VELOCITIES_KEY)
        if init_vel:
            # if user defined specifics in dictionary
            if isinstance(init_vel, dict):
                user_dict = init_vel
                # extract informatoin:
                # temperature
                temperature = user_dict.get(
                    TEMPERATURE_KEY, self.protocol.get(TEMPERATURE_KEY)
                )
                linear = user_dict.get("linear", True)
                angular = user_dict.get("angular", True)
                distribution = user_dict.get("distribution", "uniform")

            # otherwise default
            elif isinstance(init_vel, bool) and init_vel:

                temperature = self.protocol.get(TEMPERATURE_KEY)
                linear = True
                angular = True
                distribution = "uniform"

            else:
                raise KeyError(
                    f"Either set {VELOCITIES_KEY} to True or pass user requirements as dictionary."
                )

            if isinstance(temperature, list):
                temperature = temperature[0]

            vel_init = init_velocity(
                target_temperature=temperature,
                graph=self.start_graph,
                zero_linear=linear,
                zero_angular=angular,
                distribution=distribution,
                temperature_handler=self.temp,
            )

            self.start_graph[VELOCITIES_KEY] = vel_init

        # for writing to file
        # if no write frequency is given
        write_settings = self.protocol.get(WRITE_TRAJECTORY_KEY)
        self.write_freq_xyz = self.protocol.get(RUN_KEY) + 1
        self.write_freq_temp = self.protocol.get(RUN_KEY) + 1

        if write_settings:
            write_freq = write_settings.get("every", 1)

            if isinstance(write_freq, int):
                self.write_freq_xyz = write_freq
                self.write_freq_temp = write_freq

            elif isinstance(write_freq, Dict):
                self.write_freq_xyz = write_freq.get("xyz", 1)
                self.write_freq_temp = write_freq.get("temp", self.write_freq_xyz)

            else:
                raise TypeError("Write frequency must be specified as integer or dict.")
            self.save_vels = write_settings.get("save_velocities", True)
            self.filename = write_settings[FILENAME_KEY]
            self.fileformat = write_settings.get("format", "extxyz")
            logdir = os.path.dirname(self.filename)
            self.logfile = os.path.join(
                logdir, os.path.basename(self.filename).split(".")[0] + ".log"
            )
            os.makedirs(logdir, exist_ok=True)
            # write initial frame to file
            self._write_frame_to_file(
                frame=self.start_graph,
                step=0,
                append=False,
            )

    @classmethod
    def build_from_yaml(cls, filename: str):
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Could not find the file under the path {filename}"
            )

        with open(filename, "r") as file:
            dictionary = yaml.load(file, Loader=yaml.FullLoader)

        return cls(protocol=dictionary)

    def generate_trajectory(self):
        # get number of steps
        n_steps = self.protocol.get(RUN_KEY)
        # initialise frame
        frame = self.start_graph
        temp = self.temp(frame)

        with open(self.logfile, "w") as f:
            f.write("step,Temperature\n")
            f.write(f"0,{temp:3.3f}\n")

        # loop over all steps
        with torch.no_grad():
            for step in tqdm(range(1, n_steps + 1)):
                # make a step
                frame = self._make_timestep(frame, step)
                temp = self.temp(frame)

                # check for writer
                if step % self.write_freq_xyz == 0:
                    self._write_frame_to_file(
                        frame=frame,
                        step=step,
                        append=True,
                    )

                if step % self.write_freq_temp == 0:
                    with open(self.logfile, "a") as f:
                        f.write(f"{step},{temp:3.3f}\n")

    def _write_frame_to_file(
        self,
        frame: AtomicGraph,
        step: int,
        append: Optional[bool] = True,
    ):
        # get ase.Atoms object
        ase_atoms = frame.ASEAtomsObject

        if not self.save_vels:
            ase_atoms.arrays.pop("velocities")

        # add timestamp
        ase_atoms.info[FRAME_KEY] = step

        # now write to file
        ase.io.write(
            filename=self.filename,
            images=ase_atoms,
            format=self.fileformat,
            append=append,
        )

    def _make_timestep(self, frame: AtomicGraph, step: int) -> AtomicGraph:
        # let's start by calling the model and pass the graph
        frame = self.predictor(frame)

        # now let's get the predictions from the model
        model_output = torch.split(frame.target, [3, 3], dim=1)

        # now let's relate them to the fields and renormalize

        # now we update the positions based on displacements which are always given
        frame.pos = wrap_positions_back_to_box_torch(
            frame.pos + model_output[0] * self.disp_scale, frame.cell
        )
        # and same for velocities
        frame.velocities = model_output[1] * self.vel_scale

        # thermostatting
        if self.nvt:
            frame = self.thermo(frame, step)

        # manipulate momentum if required
        if self.momentum and step % self.momentum.adjust_freq == 0:
            frame = self.momentum(frame)

        # finally update edge indices
        frame.update_edge_index()

        return frame
