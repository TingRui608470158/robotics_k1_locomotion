# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to manually drive the K1 single-leg-walk velocity commands via keyboard while a trained
skrl policy checkpoint produces the joint actions, for interactive sanity testing.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard-teleop the K1 single-leg-walk velocity commands.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-K1-Single-Leg-Walk-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--v_x_sensitivity", type=float, default=0.8, help="Magnitude of forward/backward velocity command."
)
parser.add_argument("--v_y_sensitivity", type=float, default=0.4, help="Magnitude of sideways velocity command.")
parser.add_argument(
    "--omega_z_sensitivity", type=float, default=1.0, help="Magnitude of yaw rate command."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

if args_cli.headless:
    raise RuntimeError(
        "keyboard_play.py requires a live render window for keyboard input; do not pass --headless."
    )

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import numpy as np
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import k1_single_leg_walk.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


class WasdSe2Keyboard(Se2Keyboard):
    """Se2Keyboard with additional WASD (translate) + QE (yaw) bindings on top of the numpad/arrow defaults."""

    def _create_key_bindings(self):
        super()._create_key_bindings()
        self._INPUT_KEY_MAPPING.update({
            "W": np.asarray([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "A": np.asarray([0.0, 1.0, 0.0]) * self.v_y_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.v_y_sensitivity,
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.omega_z_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.omega_z_sensitivity,
        })


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Keyboard-teleop a trained skrl agent's velocity commands."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # get environment (step) dt for real-time pacing
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # configure and instantiate the skrl runner
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)

    # set up the keyboard device
    keyboard_cfg = Se2KeyboardCfg(
        v_x_sensitivity=args_cli.v_x_sensitivity,
        v_y_sensitivity=args_cli.v_y_sensitivity,
        omega_z_sensitivity=args_cli.omega_z_sensitivity,
        sim_device=str(env.unwrapped.device),
    )
    keyboard = WasdSe2Keyboard(keyboard_cfg)

    should_reset = False

    def trigger_reset():
        nonlocal should_reset
        should_reset = True

    keyboard.add_callback("R", trigger_reset)

    print(str(keyboard))
    print("\t----------------------------------------------")
    print("\tAlso available: W/S (fwd/back), A/D (left/right), Q/E (yaw), R (full env reset)")
    print("\tClose the viewport window or Ctrl+C in the terminal to quit.")

    obs, _ = env.reset()
    keyboard.reset()

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            if should_reset:
                obs, _ = env.reset()
                keyboard.reset()
                should_reset = False

            # update the velocity command from keyboard input before the agent acts on it
            cmd = keyboard.advance()  # (3,) = [v_x, v_y, omega_z]
            env.unwrapped._commands[:] = cmd.repeat(env.unwrapped.num_envs, 1)

            # agent stepping
            outputs = runner.agent.act(obs, states=obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)

        # keep the loop paced close to real time so keyboard input feels responsive
        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
