import numpy as np

import torch

from .rotation import quat_to_mat

def pose_encoding_to_extri_intri(

    pose_encoding: torch.Tensor,

    image_size_hw=None,

    pose_encoding_type: str = "absT_quaR_FoV",

    build_intrinsics: bool = True,

):

    intrinsics = None

    if pose_encoding_type != "absT_quaR_FoV":

        raise NotImplementedError

    t = pose_encoding[..., :3]

    quat = pose_encoding[..., 3:7]

    fov_h = pose_encoding[..., 7]

    fov_w = pose_encoding[..., 8]

    r = quat_to_mat(quat)

    extrinsics = torch.cat([r, t[..., None]], dim=-1)

    if build_intrinsics:

        h, w = image_size_hw

        fov_h_clamped = torch.clamp(fov_h, min=1e-6, max=np.pi - 1e-6)

        fov_w_clamped = torch.clamp(fov_w, min=1e-6, max=np.pi - 1e-6)

        fy = (h / 2.0) / torch.tan(fov_h_clamped / 2.0)

        fx = (w / 2.0) / torch.tan(fov_w_clamped / 2.0)

        fx = torch.clamp(fx, min=1.0, max=1e6)

        fy = torch.clamp(fy, min=1.0, max=1e6)

        intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)

        intrinsics[..., 0, 0] = fx

        intrinsics[..., 1, 1] = fy

        intrinsics[..., 0, 2] = w / 2

        intrinsics[..., 1, 2] = h / 2

        intrinsics[..., 2, 2] = 1.0

    return extrinsics, intrinsics
