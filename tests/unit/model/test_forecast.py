import os
import unittest

import numpy as np
import torch
from ase.io import read

from trajcast.data._keys import (
    ARCHITECTURE_KEY,
    CONFIG_KEY,
    FILENAME_KEY,
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
from trajcast.model.forecast import Forecast
from trajcast.model.forecast_tools import CSVRThermostat, Temperature
from trajcast.model.models import FlexibleModel, TrajCastModel
from trajcast.utils.atomic_computes import (
    compute_angular_momentum_for_ase_atoms,
    compute_com_velocity_for_ase_atoms,
)


class TestForecastInit(unittest.TestCase):
    def test_returns_error_if_not_allowed_fields_are_present(self):
        with self.assertRaises(KeyError, msg="Key 'fake_attr' is not allowed."):
            Forecast(protocol={"fake_attr": 2, "device": "cpu"})

    def test_returns_error_if_not_allowed_not_all_fields_are_specified(self):
        with self.assertRaises(AttributeError):
            Forecast(protocol={RUN_KEY: 2})

    def test_returns_error_if_initial_structure_not_found(self):
        with self.assertRaises(
            FileNotFoundError, msg="File with atomic coordinates not found."
        ):
            Forecast(
                protocol={
                    UNITS_KEY: "real",
                    RUN_KEY: 100,
                    TEMPERATURE_KEY: 300,
                    TIMESTEP_KEY: 10.0,
                    TYPE_MAPPER_KEY: {},
                    MODEL_KEY: "path/to/model",
                    CONFIG_KEY: "path/to/config",
                    "device": "cpu",
                }
            )

    def test_returns_error_if_model_not_found(self):
        with self.assertRaises(
            FileNotFoundError, msg="File with model parameters not found."
        ):
            Forecast(
                protocol={
                    UNITS_KEY: "real",
                    RUN_KEY: 100,
                    TEMPERATURE_KEY: 300,
                    TIMESTEP_KEY: 10.0,
                    TYPE_MAPPER_KEY: {1: 0, 8: 1, 7: 2, 6: 3},
                    MODEL_KEY: "path/to/model",
                    ARCHITECTURE_KEY: "path/to/architecture",
                    CONFIG_KEY: "tests/unit/model/data/md22_Ac-Ala3-NHMe_disp_100frames.extxyz",
                    "device": "cpu",
                }
            )

    def test_returns_error_if_architecture_missing_but_weights_given(self):
        with self.assertRaises(
            AttributeError,
            msg="Architecture needs to be provided either from file or dictionary.",
        ):
            Forecast(
                protocol={
                    UNITS_KEY: "real",
                    RUN_KEY: 100,
                    TEMPERATURE_KEY: 300,
                    TIMESTEP_KEY: 10.0,
                    TYPE_MAPPER_KEY: {},
                    MODEL_KEY: "tests/unit/model/data/forecast_benzene/model_weights.pt",
                    CONFIG_KEY: "tests/unit/model/data/md22_Ac-Ala3-NHMe_disp_100frames.extxyz",
                    "device": "cpu",
                }
            )

    def test_builds_forecaster_correctly_from_weights_and_architecture_with_config_as_dictionary_flexible_model(
        self,
    ):
        f = Forecast(
            protocol={
                UNITS_KEY: "real",
                RUN_KEY: 100,
                TEMPERATURE_KEY: 300.0,
                TIMESTEP_KEY: 10.0,
                TYPE_MAPPER_KEY: {},
                MODEL_KEY: "tests/unit/model/data/forecast_benzene/model_weights.pt",
                MODEL_TYPE_KEY: "Flexible",
                ARCHITECTURE_KEY: "tests/unit/model/data/forecast_benzene/architecture.yaml",
                CONFIG_KEY: {
                    FILENAME_KEY: "tests/unit/model/data/forecast_benzene/benzene_validation_traj.extxyz"
                },
                "device": "cpu",
            }
        )
        self.assertIsInstance(f.predictor, FlexibleModel)
        self.assertIsInstance(f.start_graph, AtomicGraph)
        self.assertAlmostEqual(f.start_graph.pos[5][1], 13.67964000)
        # check whether default values are set correctly, e.g. for removing COM velocity
        self.assertIsNone(f.momentum)

    def test_returns_error_if_momenta_are_set_but_layer_is_not_part_of_flexible_model(
        self,
    ):
        with self.assertRaises(
            AttributeError,
            msg="You cannot set the net momenta without having a 'MomentumConservation' layer to control them.",
        ):
            Forecast(
                protocol={
                    UNITS_KEY: "real",
                    RUN_KEY: 100,
                    TEMPERATURE_KEY: 300.0,
                    TIMESTEP_KEY: 10.0,
                    TYPE_MAPPER_KEY: {},
                    MODEL_KEY: "tests/unit/model/data/forecast_benzene/model_weights.pt",
                    MODEL_TYPE_KEY: "Flexible",
                    ARCHITECTURE_KEY: "tests/unit/model/data/forecast_benzene/architecture.yaml",
                    CONFIG_KEY: {
                        FILENAME_KEY: "tests/unit/model/data/forecast_benzene/benzene_validation_traj.extxyz"
                    },
                    SET_MOMENTA_KEY: {
                        "linear": torch.zeros(3),
                        "angular": torch.zeros(3),
                    },
                    "device": "cpu",
                }
            )

    def test_returns_set_momenta_are_passed_to_flexible_model_correctly(self):
        f = Forecast(
            protocol={
                UNITS_KEY: "real",
                RUN_KEY: 100,
                TEMPERATURE_KEY: 300.0,
                TIMESTEP_KEY: 10.0,
                TYPE_MAPPER_KEY: {},
                ARCHITECTURE_KEY: "tests/unit/model/data/forecast_benzene/architecture_momentum.yaml",
                MODEL_TYPE_KEY: "Flexible",
                CONFIG_KEY: {
                    FILENAME_KEY: "tests/unit/model/data/forecast_benzene/benzene_validation_traj.extxyz"
                },
                SET_MOMENTA_KEY: {
                    "linear": torch.zeros(3),
                    "angular": torch.zeros(3) + 1,
                },
                "device": "cpu",
            }
        )

        # check whether this was set correctly
        self.assertTrue(
            torch.all(f.predictor.layers.Conservation.net_lin_mom == torch.zeros(3))
        )

        self.assertTrue(
            torch.all(f.predictor.layers.Conservation.net_ang_mom == torch.zeros(3) + 1)
        )

    def test_returns_set_momenta_are_passed_to_trajcast_model_correctly(self):
        f = Forecast(
            protocol={
                UNITS_KEY: "real",
                RUN_KEY: 100,
                TEMPERATURE_KEY: 300.0,
                TIMESTEP_KEY: 10.0,
                TYPE_MAPPER_KEY: {},
                ARCHITECTURE_KEY: "tests/unit/model/data/forecast_benzene/architecture_trajcast_model.yaml",
                MODEL_TYPE_KEY: "TrajCast",
                CONFIG_KEY: {
                    FILENAME_KEY: "tests/unit/model/data/forecast_benzene/benzene_validation_traj.extxyz"
                },
                SET_MOMENTA_KEY: {
                    "linear": torch.zeros(3),
                    "angular": torch.zeros(3) + 1,
                },
                "device": "cpu",
            }
        )

        self.assertTrue(isinstance(f.predictor, TrajCastModel))

        # check whether this was set correctly
        self.assertTrue(
            torch.all(f.predictor.layers.Conservation.net_lin_mom == torch.zeros(3))
        )

        self.assertTrue(
            torch.all(f.predictor.layers.Conservation.net_ang_mom == torch.zeros(3) + 1)
        )


class TestForecastGenerateTrajectory(unittest.TestCase):
    def test_returns_correctly_predicted_next_steps_benzene_no_pbc_crossing_flexible(
        self,
    ):
        f = Forecast(
            protocol={
                UNITS_KEY: "real",
                RUN_KEY: 50,
                TEMPERATURE_KEY: 300.0,
                TIMESTEP_KEY: 10.0,
                TYPE_MAPPER_KEY: {1: 0, 6: 1},
                MODEL_KEY: "tests/unit/model/data/forecast_benzene/model_params_no_pbc_crossing.pt",
                MODEL_TYPE_KEY: "Flexible",
                ARCHITECTURE_KEY: "tests/unit/model/data/forecast_benzene/architecture.yaml",
                CONFIG_KEY: {
                    FILENAME_KEY: "tests/unit/model/data/forecast_benzene/benzene_validation_traj.extxyz",
                    "index": 0,
                },
                WRITE_TRAJECTORY_KEY: {
                    FILENAME_KEY: "tests/unit/model/data/pred.xyz",
                    "every": 1,
                },
                "device": "cpu",
            }
        )

        f.generate_trajectory()

        self.assertTrue(os.path.exists("tests/unit/model/data/pred.xyz"))
        os.remove("tests/unit/model/data/pred.xyz")

    def test_returns_correctly_predicted_next_steps_benzene_no_pbc_crossing_trajcast_model(
        self,
    ):
        f = Forecast(
            protocol={
                UNITS_KEY: "real",
                RUN_KEY: 50,
                TEMPERATURE_KEY: 300.0,
                TIMESTEP_KEY: 10.0,
                TYPE_MAPPER_KEY: {1: 0, 6: 1},
                MODEL_TYPE_KEY: "TrajCast",
                ARCHITECTURE_KEY: "tests/unit/model/data/forecast_benzene/architecture_trajcast_model.yaml",
                CONFIG_KEY: {
                    FILENAME_KEY: "tests/unit/model/data/forecast_benzene/benzene_validation_traj.extxyz",
                    "index": 0,
                },
                WRITE_TRAJECTORY_KEY: {
                    FILENAME_KEY: "tests/unit/model/data/pred.xyz",
                    "every": 1,
                },
                "device": "cpu",
            }
        )

        f.generate_trajectory()

        self.assertTrue(os.path.exists("tests/unit/model/data/pred.xyz"))
        os.remove("tests/unit/model/data/pred.xyz")

    def test_returns_trajectory_with_linear_momentum_removed_default_settings_flexible(
        self,
    ):
        # Here we check that the COM velocities are removed based on the default settings
        # that means we only pass True as an argument to the keyword and the COM is removed
        # every timestep
        f = Forecast(
            protocol={
                UNITS_KEY: "real",
                RUN_KEY: 50,
                TEMPERATURE_KEY: 300.0,
                TIMESTEP_KEY: 10.0,
                TYPE_MAPPER_KEY: {1: 0, 6: 1},
                MODEL_KEY: "tests/unit/model/data/forecast_benzene/model_params_no_pbc_crossing.pt",
                MODEL_TYPE_KEY: "Flexible",
                ARCHITECTURE_KEY: "tests/unit/model/data/forecast_benzene/architecture.yaml",
                CONFIG_KEY: {
                    FILENAME_KEY: "tests/unit/model/data/forecast_benzene/benzene_validation_traj.extxyz",
                    "index": 0,
                },
                ZERO_MOMENTUM_KEY: True,
                WRITE_TRAJECTORY_KEY: {
                    FILENAME_KEY: "tests/unit/model/data/pred.xyz",
                    "every": 1,
                },
                "device": "cpu",
            }
        )

        self.assertEqual(f.momentum.adjust_freq, 1)
        self.assertTrue(f.momentum.zero_linear)
        self.assertFalse(f.momentum.zero_angular)

        f.generate_trajectory()

        self.assertTrue(os.path.exists("tests/unit/model/data/pred.xyz"))
        # let's check whether there is no linear momentum in any frame
        pred_traj = read("tests/unit/model/data/pred.xyz", index="1:")
        for frame in pred_traj:
            com_vel = np.sum(
                frame.get_masses().reshape(-1, 1) * frame.arrays["velocities"],
                axis=0,
            ) / np.sum(frame.get_masses())

            self.assertTrue(np.allclose(com_vel, 0, atol=5e-7))

        os.remove("tests/unit/model/data/pred.xyz")

    def test_returns_trajectory_with_linear_momentum_removed_every_N_steps_flexible(
        self,
    ):
        # Here we check that the COM velocities are removed every N (here 5) steps.
        # that means, rather than True we pass a dictionary with {"every": N, "linear":True} as arguym COM is removed
        # every timestep
        f = Forecast(
            protocol={
                UNITS_KEY: "real",
                RUN_KEY: 50,
                TEMPERATURE_KEY: 300.0,
                TIMESTEP_KEY: 10.0,
                TYPE_MAPPER_KEY: {1: 0, 6: 1},
                MODEL_KEY: "tests/unit/model/data/forecast_benzene/model_params_no_pbc_crossing.pt",
                MODEL_TYPE_KEY: "Flexible",
                ARCHITECTURE_KEY: "tests/unit/model/data/forecast_benzene/architecture.yaml",
                CONFIG_KEY: {
                    FILENAME_KEY: "tests/unit/model/data/forecast_benzene/benzene_validation_traj.extxyz",
                    "index": 0,
                },
                ZERO_MOMENTUM_KEY: {"linear": True, "every": 5},
                WRITE_TRAJECTORY_KEY: {
                    FILENAME_KEY: "tests/unit/model/data/pred.xyz",
                    "every": 1,
                },
                "device": "cpu",
            }
        )

        self.assertEqual(f.momentum.adjust_freq, 5)
        self.assertTrue(f.momentum.zero_linear)
        self.assertFalse(f.momentum.zero_angular)

        f.generate_trajectory()

        self.assertTrue(os.path.exists("tests/unit/model/data/pred.xyz"))
        # let's check whether there is no linear momentum in any frame
        pred_traj = read("tests/unit/model/data/pred.xyz", index="5:50:5")
        for frame in pred_traj:
            com_vel = compute_com_velocity_for_ase_atoms(frame)
            self.assertTrue(np.allclose(com_vel, 0, atol=5e-8))

        os.remove("tests/unit/model/data/pred.xyz")

    def test_returns_trajectory_with_angular_momentum_removed_every_N_steps_flexible(
        self,
    ):
        # Here we check that the angular velocities are removed every step.
        # that means, rather than True we pass a dictionary with {"every": 1, "angular": True, "linear": False}
        # as arg such that the angular momentum is removed every timestep
        f = Forecast(
            protocol={
                UNITS_KEY: "real",
                RUN_KEY: 50,
                TEMPERATURE_KEY: 300.0,
                TIMESTEP_KEY: 10.0,
                TYPE_MAPPER_KEY: {1: 0, 6: 1},
                MODEL_KEY: "tests/unit/model/data/forecast_benzene/model_params_no_pbc_crossing.pt",
                MODEL_TYPE_KEY: "Flexible",
                ARCHITECTURE_KEY: "tests/unit/model/data/forecast_benzene/architecture.yaml",
                CONFIG_KEY: {
                    FILENAME_KEY: "tests/unit/model/data/forecast_benzene/benzene_validation_traj.extxyz",
                    "index": 0,
                },
                ZERO_MOMENTUM_KEY: {"angular": True, "every": 1, "linear": False},
                WRITE_TRAJECTORY_KEY: {
                    FILENAME_KEY: "tests/unit/model/data/pred.xyz",
                    "every": 1,
                },
                "device": "cpu",
            }
        )

        self.assertEqual(f.momentum.adjust_freq, 1)
        self.assertFalse(f.momentum.zero_linear)
        self.assertTrue(f.momentum.zero_angular)

        f.generate_trajectory()

        self.assertTrue(os.path.exists("tests/unit/model/data/pred.xyz"))
        # let's check whether there is no linear momentum in any frame
        pred_traj = read("tests/unit/model/data/pred.xyz", index="1:")
        for frame in pred_traj:
            ang_vel = compute_angular_momentum_for_ase_atoms(frame)

            self.assertTrue(np.allclose(ang_vel, 0, atol=5e-5))

        os.remove("tests/unit/model/data/pred.xyz")

    def test_returns_trajectory_with_net_total_momentum_removed_every_N_steps_flexible(
        self,
    ):
        # Here we check that the COM and angular velocities are removed every step.
        # that means, rather than True we pass a dictionary with {"every": 1, "angular": True, "linear": False}
        # as arg such that the angular and linear momentum is removed every timestep
        f = Forecast(
            protocol={
                UNITS_KEY: "real",
                RUN_KEY: 50,
                TEMPERATURE_KEY: 300.0,
                TIMESTEP_KEY: 10.0,
                TYPE_MAPPER_KEY: {1: 0, 6: 1},
                MODEL_KEY: "tests/unit/model/data/forecast_benzene/model_params_no_pbc_crossing.pt",
                MODEL_TYPE_KEY: "Flexible",
                ARCHITECTURE_KEY: "tests/unit/model/data/forecast_benzene/architecture.yaml",
                CONFIG_KEY: {
                    FILENAME_KEY: "tests/unit/model/data/forecast_benzene/benzene_validation_traj.extxyz",
                    "index": 0,
                },
                ZERO_MOMENTUM_KEY: {"angular": True, "every": 1, "linear": True},
                WRITE_TRAJECTORY_KEY: {
                    FILENAME_KEY: "tests/unit/model/data/pred.xyz",
                    "every": 1,
                },
                "device": "cpu",
            }
        )

        self.assertEqual(f.momentum.adjust_freq, 1)
        self.assertTrue(f.momentum.zero_linear)
        self.assertTrue(f.momentum.zero_angular)

        f.generate_trajectory()

        self.assertTrue(os.path.exists("tests/unit/model/data/pred.xyz"))
        # let's check whether there is no linear momentum in any frame
        pred_traj = read("tests/unit/model/data/pred.xyz", index="1:")
        for frame in pred_traj:
            com_vel = compute_com_velocity_for_ase_atoms(frame)
            self.assertTrue(np.allclose(com_vel, 0, atol=5e-7))

            ang_vel = compute_angular_momentum_for_ase_atoms(frame)
            self.assertTrue(np.allclose(ang_vel, 0, atol=5e-5))

        os.remove("tests/unit/model/data/pred.xyz")


class TestCSVRThermostat(unittest.TestCase):
    def test_returns_thermostat_is_initialised_correctly(self):
        temperature = Temperature(units="real", n_atoms=32, n_extra_dofs=6)
        csvr = CSVRThermostat(
            target_temp=300.0,
            timestep=5.0,
            damping=1000.0,
            temperature_handler=temperature,
            n_steps=50,
        )

        self.assertIsInstance(csvr.temp.conv_fac, torch.Tensor)
        self.assertEqual(csvr.temp.conv_fac.shape, torch.Size([1]))

        self.assertTrue(csvr.temp._n_dofs == 90)

    def test_returns_sum_noises_are_sampled_correclty_when_ndof_odd(self):
        n_steps = 500000
        temperature = Temperature(units="real", n_atoms=162, n_extra_dofs=3)
        csvr = CSVRThermostat(
            target_temp=300.0,
            timestep=20.0,
            damping=2000,
            temperature_handler=temperature,
            n_steps=n_steps,
        )

        gamm = []
        gauss = []
        n_dofs = temperature._n_dofs
        for i in range(500000):
            # gaussian direct
            rns = torch.randn(n_dofs - 1)
            gauss.append(rns.pow(2).sum())
            gamm.append(2 * csvr.gamma_dist.sample())

        gauss = torch.stack(gauss)
        gamm = torch.stack(gamm)

        self.assertTrue(torch.isclose(gamm.mean(), gauss.mean(), atol=1e-1))
        self.assertTrue(torch.isclose(gamm.std(), gauss.std(), atol=1e-1))

    def test_returns_sum_noises_are_sampled_correclty_when_ndof_even(self):
        n_steps = 500000
        temperature = Temperature(units="real", n_atoms=22, n_extra_dofs=6)
        csvr = CSVRThermostat(
            target_temp=150.0,
            timestep=5.0,
            damping=500,
            temperature_handler=temperature,
            n_steps=n_steps,
        )

        gamm = []
        gauss = []
        n_dofs = temperature._n_dofs
        for i in range(n_steps):
            # gaussian direct
            rns = torch.randn(n_dofs - 1)
            gauss.append(rns.pow(2).sum())
            gamm.append(2 * csvr.gamma_dist.sample() + torch.randn(1).pow(2))

        gauss = torch.stack(gauss)
        gamm = torch.stack(gamm)

        self.assertTrue(torch.isclose(gamm.mean(), gauss.mean(), atol=1e-1))
        self.assertTrue(torch.isclose(gamm.std(), gauss.std(), atol=1e-1))

    def test_returns_forward_produces_rescaled_velocities(self):
        f = Forecast(
            protocol={
                UNITS_KEY: "real",
                RUN_KEY: 100,
                TEMPERATURE_KEY: 300.0,
                TIMESTEP_KEY: 10.0,
                MODEL_KEY: "tests/unit/model/data/forecast_benzene/model_weights.pt",
                MODEL_TYPE_KEY: "Flexible",
                ARCHITECTURE_KEY: "tests/unit/model/data/forecast_benzene/architecture.yaml",
                CONFIG_KEY: {
                    FILENAME_KEY: "tests/unit/data/data/aspirin_energy.extxyz"
                },
                "device": "cpu",
            }
        )

        graph = f.start_graph
        n_steps = 500000
        temperature = Temperature(units="real", n_atoms=graph.num_nodes, n_extra_dofs=0)
        csvr = CSVRThermostat(
            target_temp=150.0,
            timestep=5.0,
            damping=50.0,
            temperature_handler=temperature,
            n_steps=n_steps,
        )
        temps = []
        for step in range(n_steps):
            graph = csvr(graph, step)
            T = temperature(graph)
            temps.append(T)

        temps = torch.tensor(temps[10000:])

        expected_std = (2 / 3 / graph.num_nodes * 150**2) ** 0.5

        self.assertTrue(
            torch.isclose(torch.tensor([expected_std]), temps.std(), atol=1e-1)
        )

        self.assertTrue(torch.isclose(torch.tensor([150.0]), temps.mean(), atol=5 - 2))

    def test_returns_thermostatting_is_reproducible_via_seed(self):

        f = Forecast(
            protocol={
                UNITS_KEY: "real",
                RUN_KEY: 100,
                TEMPERATURE_KEY: 300.0,
                TIMESTEP_KEY: 10.0,
                MODEL_KEY: "tests/unit/model/data/forecast_benzene/model_weights.pt",
                MODEL_TYPE_KEY: "Flexible",
                ARCHITECTURE_KEY: "tests/unit/model/data/forecast_benzene/architecture.yaml",
                CONFIG_KEY: {
                    FILENAME_KEY: "tests/unit/data/data/aspirin_energy.extxyz"
                },
                "seed": 42,
                VELOCITIES_KEY: {
                    TEMPERATURE_KEY: 373.15,
                    "linear": True,
                    "angular": True,
                    "distribution": "uniform",
                },
            }
        )
        n_steps = 500
        graph = f.start_graph
        csvr = CSVRThermostat(
            target_temp=250.0,
            timestep=5.0,
            damping=500.0,
            temperature_handler=f.temp,
            n_steps=n_steps,
        )

        temps_s1 = []
        vel_s1 = []
        for steps in range(n_steps):
            T = f.temp(graph)
            vel_s1.append(graph[VELOCITIES_KEY][0])
            graph = csvr(graph, steps)

            temps_s1.append(T)

        temps_s1 = torch.tensor(temps_s1)
        vel_s1 = torch.stack(vel_s1)
        # check the initial velocity is correct
        self.assertTrue(
            np.isclose(
                temps_s1[0].item(),
                373.15,
            )
        )

        # seed 2
        f = Forecast(
            protocol={
                UNITS_KEY: "real",
                RUN_KEY: 100,
                TEMPERATURE_KEY: 300.0,
                TIMESTEP_KEY: 10.0,
                MODEL_KEY: "tests/unit/model/data/forecast_benzene/model_weights.pt",
                MODEL_TYPE_KEY: "Flexible",
                ARCHITECTURE_KEY: "tests/unit/model/data/forecast_benzene/architecture.yaml",
                CONFIG_KEY: {
                    FILENAME_KEY: "tests/unit/data/data/aspirin_energy.extxyz"
                },
                "seed": 42,
                VELOCITIES_KEY: {
                    TEMPERATURE_KEY: 373.15,
                    "linear": True,
                    "angular": True,
                    "distribution": "uniform",
                },
            }
        )

        graph = f.start_graph

        # self.assertTrue(torch.all(torch.equal(v1, v2)))

        temps_s2 = []
        vel_s2 = []
        for steps in range(500):
            T = f.temp(graph)
            vel_s2.append(graph[VELOCITIES_KEY][0])
            graph = csvr(graph, steps)

            temps_s2.append(T)

        temps_s2 = torch.tensor(temps_s2)
        vel_s2 = torch.stack(vel_s2)
        self.assertTrue(torch.allclose(temps_s2, temps_s1))
        self.assertTrue(torch.allclose(vel_s2, vel_s1))

    def test_returns_csvr_is_properly_initialised_within_forecast(self):
        f = Forecast(
            protocol={
                UNITS_KEY: "real",
                RUN_KEY: 100,
                TEMPERATURE_KEY: 300.0,
                TIMESTEP_KEY: 10.0,
                MODEL_KEY: "tests/unit/model/data/forecast_benzene/model_weights.pt",
                MODEL_TYPE_KEY: "Flexible",
                ARCHITECTURE_KEY: "tests/unit/model/data/forecast_benzene/architecture.yaml",
                CONFIG_KEY: {
                    FILENAME_KEY: "tests/unit/data/data/aspirin_energy.extxyz"
                },
                "seed": 42,
                THERMOSTAT_KEY: {"Tdamp": 1000.0},
            }
        )

        self.assertTrue(hasattr(f, "thermo"))
        self.assertIsInstance(f.thermo, CSVRThermostat)
        self.assertTrue(f.temp._n_extra_dofs == 0)


if __name__ == "__main__":
    unittest.main()
