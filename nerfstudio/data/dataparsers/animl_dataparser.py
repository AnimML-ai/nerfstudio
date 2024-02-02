# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import open3d as o3d
import torch
from PIL import Image

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE

MAX_AUTO_RESOLUTION = 1600


@dataclass
class AnimlDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: Animl)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    load_depth: bool = False
    """Whether to load depth data"""
    load_mask: bool = True
    """Whether to load mask data"""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    downscale_factor: Literal[1, 2] = 2
    """ Use half resolution image or full resolution"""

@dataclass
class Animl(DataParser):
    """Nerfstudio DatasetParser"""

    config: AnimlDataParserConfig
    

    def _generate_dataparser_outputs(self, split="train"):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        frames = load_from_json(self.config.data / "transforms.json")
        info = load_from_json(self.config.data / "info.json")
        camera_center = np.array(info["cameraCenter"])
        camera_radius = info["cameraSphereRadius"]
        data_dir = self.config.data

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame_id, frame in frames.items():
            image_dir_name = "images"
            if self.config.downscale_factor == 2:
                image_dir_name = "images_2"
            fname = data_dir / image_dir_name / f"{frame_id}.jpg"
            K = frame["m_intr_triangulated"]
            fx.append(K[0][0])
            fy.append(K[1][1])
            cx.append(K[0][2])
            cy.append(K[1][2])
            height.append(frame["height"])
            width.append(frame["width"])
            distort.append(camera_utils.get_distortion_params())


            image_filenames.append(fname)
            pose = np.array(frame["m_extr_triangulated"])
            pose[:3, 3] -= camera_center
            pose[:3, 3] /= camera_radius
            pose = pose[[0, 2, 1, 3], :]
            pose[1, :] *= -1
            pose[3, 3] = 1.0 
            poses.append(pose)
            
            if self.config.load_mask:
                mask_filenames.append(data_dir / "aot_masks" / f"{frame_id}.jpg")

            if self.config.load_depth:
                depth_filenames.append(data_dir / "lidar_depth" / f"{frame_id}.png")
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        assert len(mask_filenames) == 0 or (len(mask_filenames) == len(image_filenames)), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (len(depth_filenames) == len(image_filenames)), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        i_train, i_eval = get_train_eval_split_fraction(image_filenames, 1.0)
        
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")


        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = 1.0
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        camera_type = CameraType.PERSPECTIVE

        fx = torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type
        )

        cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)


        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=1.0,
            dataparser_transform=torch.eye(4),
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "alpha_filenames": mask_filenames if len(mask_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor
            },
        )
        return dataparser_outputs