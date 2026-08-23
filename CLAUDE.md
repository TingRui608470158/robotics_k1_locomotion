# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an external Isaac Lab extension project (built from the Isaac Lab extension template) that trains a K1 humanoid robot to walk on a single leg using RL. It is developed outside the core Isaac Lab repository and installed in editable mode into a Python environment that already has Isaac Lab installed. The task uses Isaac Lab's Direct RL workflow (not the manager-based workflow). Both skrl and RSL-RL are wired up as RL libraries (separate `scripts/skrl/` and `scripts/rsl_rl/` entry points registered against the same gym env id); skrl is what's been used/tuned so far.

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

Train/play with RSL-RL instead of skrl (same task, separate entry point — requires `rsl-rl-lib` installed; not yet tuned, `skrl_ppo_cfg.yaml`-equivalent hyperparameters were ported over but never validated against a training run):
```bash
python scripts/rsl_rl/train.py --task=Template-K1-Single-Leg-Walk-Direct-v0
python scripts/rsl_rl/play.py --task=Template-K1-Single-Leg-Walk-Direct-v0 --checkpoint=<path>
```
RSL-RL logs go to `logs/rsl_rl/k1_train_rsl_rl/` (skrl logs go to `logs/skrl/k1_train/`). RSL-RL natively supports recurrent (LSTM/GRU) policies via `RslRlPpoActorCriticRecurrentCfg` (just a config swap, no custom model code needed) — unlike skrl's declarative `network: layers:` YAML style, which only builds stateless feedforward stacks and has no RNN support without writing a custom model class.

`scripts/curve.py` is a standalone (non-Isaac-Lab) matplotlib helper that predates the arXiv:2404.19173 reward rewrite — it visualizes the old phase-clock cubic-Bézier `swing_height` profile (`_expected_foot_height()`), which no longer exists in `reward_terms.py`. It's currently orphaned; either delete it or repoint it at something in the current reward set before relying on it.

Useful `train.py`/`play.py`/`keyboard_play.py` flags: `--num_envs`, `--seed`, `--headless` (from `AppLauncher`, not valid for `keyboard_play.py`), `--video`, `--checkpoint`, `--max_iterations`. skrl-only: `--algorithm` (defaults to `PPO`), `--ml_framework` (`torch`/`jax`/`jax-numpy`). RSL-RL-only: `--resume`, `--load_run`, `--experiment_name`, `--run_name`, `--logger`.

Lint/format (ruff via pre-commit):
```bash
pip install pre-commit
pre-commit run --all-files
```

There are no unit tests in this repo currently; correctness is validated by running training/play/zero/random agent scripts against Isaac Sim.

## Architecture

- `source/k1_single_leg_walk/` is the installable Python package (`k1_single_leg_walk`), registered as an Isaac Lab extension via `config/extension.toml`. `setup.py` reads package metadata from `extension.toml`.
- `k1_single_leg_walk/tasks/direct/k1_single_leg_walk/` is the task implementation, following Isaac Lab's Direct RL env pattern:
  - `k1_single_leg_walk_env_cfg.py` — `K1SingleLegWalkEnvCfg(DirectRLEnvCfg)`: sim/scene/terrain setup (flat-plane `TerrainImporterCfg`), robot config (`K1_HUMANOID_CFG` from `isaaclab_assets.robots.Robotics_K1`), the `joint_action_scale_map` (per joint-group `(action_scale, max_delta)` in radians — action_scale must be >= max_delta for the clamp to actually bind), `arm_joint_keys` (which joint groups count as "arm" for the loose arm-deviation reward — legs/waist are intentionally excluded), termination thresholds (`min_torso_height`, `max_torso_tilt`), command-sampling ranges (`min/max_vx`, `min/max_vy`, `min/max_yaw_rate` — matching the arXiv:2404.19173 `cu=[cx,cy,cyaw]` ranges, `cx` intentionally asymmetric — plus `min/max_resample_interval_s`), training-time push-perturbation settings (`enable_random_push` — defaults to `False`; `random_push_prob`, `min/max_push_force`, also from the paper but disabled by default after it produced confusing "moves without command" symptoms during interactive testing — both `play.py` and `keyboard_play.py` additionally force it off regardless of this default), and all reward scale/coefficient hyperparameters — including `origin_height`/`contact_height_threshold` (a fixed constant, not a curriculum), which define ground contact geometrically (see below).
  - `k1_single_leg_walk_env.py` — `K1SingleLegWalkEnv(DirectRLEnv)`: implements `_setup_scene`, `_pre_physics_step` (maps actions through `joint_action_scale_map` to a clamped delta added to `default_joint_pos`), `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx`. Only joints present in `joint_action_scale_map` are controlled (matched by substring against `robot.data.joint_names`); everything else stays at its default pose. There is no `ContactSensor` — "foot in contact" is purely geometric: `ankle_roll_link` world z minus `origin_height` (the ankle-to-sole offset) compared against the fixed `contact_height_threshold` cfg constant, so that threshold doubles as a de-facto minimum swing-clearance requirement. `_get_rewards` maintains the contact state plus a hand-rolled per-foot air-time timer (`_foot_air_time`/`_prev_foot_in_contact`, since there's no sensor to track it) and a short rolling history buffer (`single_contact_grace_period` long) used by the single-foot-contact reward. The yaw command is a *rate* (`_yaw_rate_cmd`, matching the paper's `cyaw`), not a one-shot absolute heading: `_get_rewards()` integrates it into `_commands[:, 2]` (the actual heading target `yaw_tracking_reward` tracks via `quat_dist`) every step, wrapped to `[-pi, pi]`; a "stand still" episode just samples a zero rate, freezing the heading at the post-reset orientation. When `debug_vis` is on, green arrow markers show the commanded root velocity (or a fixed-length arrow toward the current heading target while standing, since a zero-velocity arrow would otherwise vanish).
  - `__init__.py` registers the gym env id `Template-K1-Single-Leg-Walk-Direct-v0`, pointing at the env cfg plus both the skrl and RSL-RL agent cfg entry points.
  - `agents/skrl_ppo_cfg.yaml` — skrl PPO agent/model/memory/trainer configuration (network architecture, PPO hyperparameters, seed). `agents/rsl_rl_ppo_cfg.py` — the RSL-RL equivalent (`K1SingleLegWalkPPORunnerCfg`), hyperparameters ported from the skrl yaml for rough parity but not independently tuned.
- Observations (85-dim) are: root lin vel (3) + root ang vel (3) + projected gravity (3) + command `(vx, vy, yaw_target)` (3) + controlled-joint pos deviation from default (23) + controlled-joint vel (23) + last action (23) + heading-error sin/cos (2) + per-foot contact state (2).
- Reward design follows *Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking* (arXiv:2404.19173); all 14 terms live in `reward_terms.py` (see its module docstring for the full rationale) and are bounded, non-negative `exp(-coeff*error)`-style kernels (or a 0/1 gate times a kernel) multiplied by a small positive scale, then summed in `_get_rewards()` and scaled by `step_dt`:
  - **Core command tracking** (xy linear-velocity tracking — squared-error exp kernel normally, a linear-in-speed kernel when the sampled command is "stand still"; yaw-heading tracking; roll/pitch-upright tracking) — using these three alone is known to produce a degenerate two-footed hopping gait that still satisfies the commands.
  - **Single foot contact** (`single_foot_contact_reward`) replaces the old phase-clock/foot-height-tracking mechanism entirely as the fix for that hopping gait; it's the most reliable of several approaches tried in the paper.
  - **Leg symmetry** and **stand-still** (`leg_symmetry_reward`, `stand_still_reward`) are *not* from the paper — added after observing two exploits during training: (1) `single_foot_contact_reward` only checks "exactly one foot down," not *which* foot, so a policy can permanently drag one foot on the ground and only ever lift the other, never learning a real alternating gait — `leg_symmetry_reward` penalizes an EMA'd per-foot contact-time-fraction imbalance (`leg_symmetry_tau`) to close that loophole. (2) nothing penalized fidgeting during a "stand still" command (`single_foot_contact`/`feet_position` don't care about motion, only position/contact-count) — `stand_still_reward` directly rewards near-zero controlled-joint velocity, gated to `is_standing`.
  - **Style / sim-to-real auxiliary terms**: base height, feet air time (the only sparse term, hence its high 1.0 scale — gives a flat 1.0 while standing rather than 0, unlike a plain gate), feet orientation (gate is "is there a nonzero `cyaw` command", not measured heading error; checks roll+pitch only while turning, +yaw alignment too while going straight), feet position (tracked only while standing; flat 1.0 otherwise, not 0), arm deviation (arms/wrists only, not legs/waist), base acceleration (linear only, no angular), action difference, torque (normalized per-joint by `joint_effort_limits` before averaging, not raw Nm).
  - All coefficients/formulas now come from the paper's Reward Term Definition/Weighting table (transcribed directly, including which terms use linear vs. squared error inside the exp — most auxiliary terms are linear, not squared) — see `reward_terms.py`'s module docstring for the one exception (`feet_orientation`'s coefficient was illegible in the OCR'd table; a placeholder is used and flagged in-code). Episode reward sums are logged per-term to TensorBoard under `Episode_Reward/*` on reset. Episodes terminate on falling below `min_torso_height` or tilting past `max_torso_tilt` (angle derived from `projected_gravity_b`).
- **Command sampling** (`K1SingleLegWalkEnv._sample_commands()`) also follows the paper's protocol, not just its `cu=[cx,cy,cyaw]` ranges: every env independently draws one of 5 categories uniformly — stand (all axes zero), sagittal (`vx` only), lateral (`vy` only), turn-in-place (yaw rate only), omnidirectional (all three) — via `_command_axes_active`, a `(5, 3)` mask table indexed by the sampled category. This happens on reset *and* mid-episode: each env independently counts down `_steps_until_resample` (redrawn each time to a random `[min_resample_interval_s, max_resample_interval_s]` value) and calls `_sample_commands()` again the moment it hits zero, so a single 16 s episode (`episode_length_s`) will typically see several different commands, not just the one sampled at reset. `is_standing` (drives the standing-mode branches throughout `reward_terms.py`) is derived from the sampled category and updates along with it.
- **Training-time push perturbations** (`K1SingleLegWalkEnv._sample_push()`/`_apply_action()`), also from the paper, but **disabled by default** (`enable_random_push = False`) after it caused confusing behavior during interactive `play.py`/`keyboard_play.py` testing — set `enable_random_push = True` to re-enable for training. When enabled: every `env.step()`, each env independently has a `random_push_prob` (1%) chance of a random-horizontal-direction force in `[min_push_force, max_push_force]` N being applied to the pelvis (root body) for that whole step. It's re-applied every physics substep in `_apply_action()` via `robot.instantaneous_wrench_composer` (which only lasts one physics step per call, unlike the deprecated `set_external_force_and_torque`/`permanent_wrench_composer`, so it must be refreshed each substep) — this makes the force act for the full `step_dt` without any manual clearing needed, since the next step's `_sample_push()` (almost always zero) naturally overwrites it. `_sample_push()` short-circuits to an all-zero force whenever `enable_random_push` is `False`. Both `play.py` and `keyboard_play.py` force `enable_random_push = False` on the env cfg regardless of the training-time default, so evaluation/teleop never sees pushes even if training was run with it enabled.
- `scripts/list_envs.py`, `scripts/zero_agent.py`, `scripts/random_agent.py` are generic Isaac Lab template scripts that work against any task registered under the `Template-*` gym id pattern (this pattern is defined in `list_envs.py` and must be updated if the task name prefix changes).
- Comments in the env/cfg files are in Traditional Chinese; keep that convention when editing reward/action logic in those files.

## Code style

- Ruff is configured in `pyproject.toml` (line length 120, target py310) with a custom isort section order that separates Isaac Lab/Omniverse/first-party imports (see `[tool.ruff.lint.isort.sections]`). Follow this import grouping in new files: stdlib → third-party → omniverse (`isaacsim`, `omni`, `pxr`, `carb`, ...) → `isaaclab` → `isaaclab_contrib`/`isaaclab_rl`/`isaaclab_mimic`/`isaaclab_tasks`/`isaaclab_assets` → local.
- Pre-commit also inserts an SPDX/copyright license header (`.github/LICENSE_HEADER.txt`) on `.py`/`.yml`/`.yaml` files — keep the existing header block at the top of files intact.
- Pyright type checking mode is "basic"; missing-import and general type-issue diagnostics are relaxed project-wide (see `[tool.pyright]` in `pyproject.toml`) since Isaac Sim/Omniverse modules aren't installed in the lint environment.
