# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an external Isaac Lab extension project (built from the Isaac Lab extension template) that trains a K1 humanoid robot to walk on a single leg using RL. It is developed outside the core Isaac Lab repository and installed in editable mode into a Python environment that already has Isaac Lab installed. The task uses Isaac Lab's Direct RL workflow (not the manager-based workflow) and skrl as the RL library.

## Setup and commands

All commands must be run with a Python interpreter that has Isaac Lab installed (conda/uv env, or use `ISAACLAB_PATH/isaaclab.sh|bat -p python` in place of `python`).

Install the extension in editable mode:
```bash
python -m pip install -e source/k1_single_leg_walk
```

List available tasks (registered gym env ids):
```bash
python scripts/list_envs.py
```

Train with skrl PPO:
```bash
python scripts/skrl/train.py --task=Template-K1-Single-Leg-Walk-Direct-v0
```

Play/evaluate a trained checkpoint:
```bash
python scripts/skrl/play.py --task=Template-K1-Single-Leg-Walk-Direct-v0 --checkpoint=<path>
```

Sanity-check the environment without a policy:
```bash
python scripts/zero_agent.py --task=Template-K1-Single-Leg-Walk-Direct-v0
python scripts/random_agent.py --task=Template-K1-Single-Leg-Walk-Direct-v0
```

Interactively drive a trained checkpoint's velocity command with the keyboard (WASD move, Q/E yaw, R to reset; requires a live render window, so never pass `--headless`):
```bash
python scripts/skrl/keyboard_play.py --task=Template-K1-Single-Leg-Walk-Direct-v0 --checkpoint=<path>
```

`scripts/curve.py` is a standalone (non-Isaac-Lab) matplotlib helper for visually tuning `swing_height` against the cubic-Bézier expected-foot-height profile used in `_expected_foot_height()`/the foot-height reward — run it directly with a plain `python` interpreter, no Isaac Sim needed.

Useful `train.py`/`play.py`/`keyboard_play.py` flags: `--num_envs`, `--seed`, `--headless` (from `AppLauncher`, not valid for `keyboard_play.py`), `--video`, `--checkpoint`, `--max_iterations`, `--algorithm` (defaults to `PPO`), `--ml_framework` (`torch`/`jax`/`jax-numpy`).

Lint/format (ruff via pre-commit):
```bash
pip install pre-commit
pre-commit run --all-files
```

There are no unit tests in this repo currently; correctness is validated by running training/play/zero/random agent scripts against Isaac Sim.

## Architecture

- `source/k1_single_leg_walk/` is the installable Python package (`k1_single_leg_walk`), registered as an Isaac Lab extension via `config/extension.toml`. `setup.py` reads package metadata from `extension.toml`.
- `k1_single_leg_walk/tasks/direct/k1_single_leg_walk/` is the task implementation, following Isaac Lab's Direct RL env pattern:
  - `k1_single_leg_walk_env_cfg.py` — `K1SingleLegWalkEnvCfg(DirectRLEnvCfg)`: sim/scene/terrain setup (flat-plane `TerrainImporterCfg`), robot config (`K1_HUMANOID_CFG` from `isaaclab_assets.robots.Robotics_K1`), the `joint_action_scale_map` (per joint-group `(action_scale, max_delta)` in radians — action_scale must be >= max_delta for the clamp to actually bind), `pose_weights_map` (per joint-group weight on the default-pose deviation penalty — legs are weighted low since walking requires deviating from default, wrists high since they should stay near-locked), termination thresholds (`min_torso_height`, `max_torso_tilt`), command-sampling ranges (`min/max_lin_speed`, `min/max_ang_speed`, `stand_still_prob`), and all reward scale/kernel hyperparameters.
  - `k1_single_leg_walk_env.py` — `K1SingleLegWalkEnv(DirectRLEnv)`: implements `_setup_scene`, `_pre_physics_step` (maps actions through `joint_action_scale_map` to a clamped delta added to `default_joint_pos`, and advances the per-foot gait phase), `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx`. Only joints present in `joint_action_scale_map` are controlled (matched by substring against `robot.data.joint_names`); everything else stays at its default pose. Maintains a 2-element gait phase per env (left/right foot, right offset by π so they alternate) advanced each physics step by `_phase_dt` (derived from `gait_cycle_time`); `_expected_foot_height()` turns phase into a target foot height via a cubic-Bézier swing/stance profile (visualized/tunable in `scripts/curve.py`) that the foot-height reward tracks. When `debug_vis` is on, green/blue arrow markers show commanded vs. actual root velocity.
  - `__init__.py` registers the gym env id `Template-K1-Single-Leg-Walk-Direct-v0`, pointing at both the env cfg and the skrl agent cfg entry points.
  - `agents/skrl_ppo_cfg.yaml` — skrl PPO agent/model/memory/trainer configuration (network architecture, PPO hyperparameters, seed).
- Observations (85-dim) are: root lin vel (3) + root ang vel (3) + projected gravity (3) + velocity command (3) + controlled-joint pos deviation from default (23) + controlled-joint vel (23) + last action (23) + per-foot gait phase sin (2) + per-foot gait phase cos (2).
- Reward is a weighted sum of 9 scale terms grouped into 7 numbered categories in `_get_rewards()` (velocity tracking; foot-height gait tracking; weighted joint-deviation-from-default penalty via `pose_weights_map`; feet orientation penalty + feet-too-close penalty; alive bonus; torso-upright penalty + torso angular-velocity penalty; action-rate penalty), each with its own scale in the env cfg; exponential-kernel terms use `*_std`/`*_sigma` config fields, and the total is scaled by `step_dt`. Episode reward sums are logged per-term to TensorBoard under `Episode_Reward/reward/*` on reset. Episodes terminate on falling below `min_torso_height` or tilting past `max_torso_tilt` (angle derived from `projected_gravity_b`); on reset, commands are sampled as a random heading + speed in `[min_lin_speed, max_lin_speed]` for lin x/y and `[min_ang_speed, max_ang_speed]` for yaw rate, with `stand_still_prob` chance of a zero ("stand still") command.
- `scripts/list_envs.py`, `scripts/zero_agent.py`, `scripts/random_agent.py` are generic Isaac Lab template scripts that work against any task registered under the `Template-*` gym id pattern (this pattern is defined in `list_envs.py` and must be updated if the task name prefix changes).
- Comments in the env/cfg files are in Traditional Chinese; keep that convention when editing reward/action logic in those files.

## Code style

- Ruff is configured in `pyproject.toml` (line length 120, target py310) with a custom isort section order that separates Isaac Lab/Omniverse/first-party imports (see `[tool.ruff.lint.isort.sections]`). Follow this import grouping in new files: stdlib → third-party → omniverse (`isaacsim`, `omni`, `pxr`, `carb`, ...) → `isaaclab` → `isaaclab_contrib`/`isaaclab_rl`/`isaaclab_mimic`/`isaaclab_tasks`/`isaaclab_assets` → local.
- Pre-commit also inserts an SPDX/copyright license header (`.github/LICENSE_HEADER.txt`) on `.py`/`.yml`/`.yaml` files — keep the existing header block at the top of files intact.
- Pyright type checking mode is "basic"; missing-import and general type-issue diagnostics are relaxed project-wide (see `[tool.pyright]` in `pyproject.toml`) since Isaac Sim/Omniverse modules aren't installed in the lint environment.
