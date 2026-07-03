import numpy as np

import torch

def closed_form_inverse_se3(se3, r=None, t=None):

    is_numpy = isinstance(se3, np.ndarray)

    if se3.shape[-2:] not in {(4, 4), (3, 4)}:

        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    if r is None:

        r = se3[:, :3, :3]

    if t is None:

        t = se3[:, :3, 3:]

    if is_numpy:

        r_t = np.transpose(r, (0, 2, 1))

        top_right = -np.matmul(r_t, t)

        inverted = np.tile(np.eye(4), (len(r), 1, 1))

    else:

        r_t = r.transpose(1, 2)

        top_right = -torch.bmm(r_t, t)

        inverted = torch.eye(4, 4, device=r.device, dtype=r.dtype)[None].repeat(len(r), 1, 1)

    inverted[:, :3, :3] = r_t

    inverted[:, :3, 3:] = top_right

    return inverted

def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:

    h, w = depth_map.shape

    fu, fv = intrinsic[0, 0], intrinsic[1, 1]

    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    fu = max(float(fu), 1e-6)

    fv = max(float(fv), 1e-6)

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    x_cam = (u - cu) * depth_map / fu

    y_cam = (v - cv) * depth_map / fv

    z_cam = depth_map

    return np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

def depth_to_world_coords_points(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray, eps=1e-8):

    if depth_map is None:

        return None, None, None

    point_mask = depth_map > eps

    cam_coords = depth_to_cam_coords_points(depth_map, intrinsic)

    cam_to_world = closed_form_inverse_se3(extrinsic[None])[0]

    r_cam_to_world = cam_to_world[:3, :3]

    t_cam_to_world = cam_to_world[:3, 3]

    world_coords = np.dot(cam_coords, r_cam_to_world.T) + t_cam_to_world

    return world_coords, cam_coords, point_mask

def unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam) -> np.ndarray:

    if isinstance(depth_map, torch.Tensor):

        depth_map = depth_map.cpu().numpy()

    if isinstance(extrinsics_cam, torch.Tensor):

        extrinsics_cam = extrinsics_cam.cpu().numpy()

    if isinstance(intrinsics_cam, torch.Tensor):

        intrinsics_cam = intrinsics_cam.cpu().numpy()

    world_points_list = []

    for frame_idx in range(depth_map.shape[0]):

        cur_world_points, _, _ = depth_to_world_coords_points(

            depth_map[frame_idx].squeeze(-1), extrinsics_cam[frame_idx], intrinsics_cam[frame_idx]

        )

        world_points_list.append(cur_world_points)

    return np.stack(world_points_list, axis=0)
