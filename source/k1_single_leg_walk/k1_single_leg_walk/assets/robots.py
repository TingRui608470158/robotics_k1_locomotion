# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Mujoco Humanoid robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

##
# Configuration
##
K1_HUMANOID_CFG = ArticulationCfg(
    actuator_value_resolution_debug_print = True,
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.join(_THIS_DIR, "ai_sapiens_description/urdf", "k1.urdf"),
        usd_dir=os.path.join(_THIS_DIR, "ai_sapiens_description/usd"),
        usd_file_name="k1.usd",
        force_usd_conversion=True,

        fix_base=False,
        activate_contact_sensors=True,
        
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,   # ← 加這行
        ),

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,      # ★ 這行是關鍵
        ),
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness=100.0,
                damping=10.0,
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
                    ".*elbow_joint":1.3 ,
                    ".*hip_pitch_joint": -0.2,   # 髖部微彎（負值視你的關節方向定義而定）
                    ".*knee_joint": 0.7,         # 膝蓋明顯彎曲（正值，彎曲角度比之前的 0.3 再大一些）
                    ".*ankle_pitch_joint": -0.3, # 腳踝反向補償，讓小腿/軀幹角度抵銷回垂直
                    ".*shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 0.3,
                    "right_shoulder_roll_joint": -0.3,
                    },
    ),
    actuators={
        # 腿部：承重最多，需要較高的 stiffness/damping
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*hip_pitch_joint",
                ".*hip_roll_joint",
                ".*hip_yaw_joint",
                ".*knee_joint",
            ],
            stiffness=300.0,
            damping=30.0,
        ),
        # 腳踝：需要精細控制平衡，但負載相對小
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*ankle_pitch_joint",
                ".*ankle_roll_joint",
            ],
            stiffness=200.0,
            damping=20.0,
            # effort_limit=550.0,
        ),
        # 腰部：連接上下半身，中等剛性
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            stiffness=100.0,
            damping=10.0,
        ),
        # 手臂：肩膀+手肘，較輕負載
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*shoulder_pitch_joint",
                ".*shoulder_roll_joint",
                ".*shoulder_yaw_joint",
                ".*elbow_joint",
            ],
            stiffness=100.0,
            damping=10.0,
        ),
        # 手腕：最末端、最輕
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[".*wrist_roll_joint"],
            stiffness=100.0,
            damping=10.0,
        ),
    },
    
)
"""Configuration for the Mujoco Humanoid robot."""
