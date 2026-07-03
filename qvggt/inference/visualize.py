import numpy as np

from scipy.spatial.transform import Rotation

def _require_trimesh():

    try:

        import trimesh

    except ModuleNotFoundError as exc:

        raise ModuleNotFoundError(

            "Visualization export requires `trimesh`. Install it from requirements.txt "

            "or run inference without `--export-glb`."

        ) from exc

    return trimesh

def predictions_to_glb(

    predictions: dict,

    conf_thres: float = 3.0,

    filter_by_frames: str = "all",

    mask_black_bg: bool = False,

    mask_white_bg: bool = False,

    show_cam: bool = True,

    prediction_mode: str = "pointmap",

):

    trimesh = _require_trimesh()

    if prediction_mode == "depth":

        pred_world_points = predictions["world_points_from_depth"]

        pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    else:

        pred_world_points = predictions.get("world_points", predictions["world_points_from_depth"])

        pred_world_points_conf = predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))

    images = predictions["images"]

    camera_matrices = predictions["extrinsic"]

    selected_frame_idx = None

    if filter_by_frames not in {"all", "All"}:

        try:

            selected_frame_idx = int(str(filter_by_frames).split(":")[0])

        except (ValueError, IndexError):

            selected_frame_idx = None

    if selected_frame_idx is not None:

        pred_world_points = pred_world_points[selected_frame_idx][None]

        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]

        images = images[selected_frame_idx][None]

        camera_matrices = camera_matrices[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)

    if images.ndim == 4 and images.shape[1] == 3:

        colors_rgb = np.transpose(images, (0, 2, 3, 1))

    else:

        colors_rgb = images

    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    conf = pred_world_points_conf.reshape(-1)

    conf_threshold = 0.0 if conf_thres == 0.0 else np.percentile(conf, conf_thres)

    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    if mask_black_bg:

        conf_mask &= colors_rgb.sum(axis=1) >= 16

    if mask_white_bg:

        conf_mask &= ~((colors_rgb[:, 0] > 240) & (colors_rgb[:, 1] > 240) & (colors_rgb[:, 2] > 240))

    vertices_3d = vertices_3d[conf_mask]

    colors_rgb = colors_rgb[conf_mask]

    if vertices_3d.size == 0:

        vertices_3d = np.array([[1, 0, 0]])

        colors_rgb = np.array([[255, 255, 255]])

        scene_scale = 1

    else:

        lower = np.percentile(vertices_3d, 5, axis=0)

        upper = np.percentile(vertices_3d, 95, axis=0)

        scene_scale = np.linalg.norm(upper - lower)

    scene_3d = trimesh.Scene()

    scene_3d.add_geometry(trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb))

    num_cameras = len(camera_matrices)

    extrinsics_matrices = np.zeros((num_cameras, 4, 4))

    extrinsics_matrices[:, :3, :4] = camera_matrices

    extrinsics_matrices[:, 3, 3] = 1

    if show_cam:

        cmap = __import__("matplotlib").colormaps.get_cmap("gist_rainbow")

        for i in range(num_cameras):

            world_to_camera = extrinsics_matrices[i]

            camera_to_world = np.linalg.inv(world_to_camera)

            rgba_color = cmap(i / num_cameras)

            current_color = tuple(int(255 * x) for x in rgba_color[:3])

            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)

    return apply_scene_alignment(scene_3d, extrinsics_matrices)

def integrate_camera_into_scene(scene, transform: np.ndarray, face_colors: tuple, scene_scale: float):

    trimesh = _require_trimesh()

    cam_width = scene_scale * 0.05

    cam_height = scene_scale * 0.1

    rot_45_degree = np.eye(4)

    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()

    rot_45_degree[2, 3] = -cam_height

    complete_transform = transform @ get_opengl_conversion_matrix() @ rot_45_degree

    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    slight_rotation = np.eye(4)

    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices_combined = np.concatenate(

        [

            camera_cone_shape.vertices,

            0.95 * camera_cone_shape.vertices,

            transform_points(slight_rotation, camera_cone_shape.vertices),

        ]

    )

    vertices_transformed = transform_points(complete_transform, vertices_combined)

    mesh_faces = compute_camera_faces(camera_cone_shape)

    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)

    camera_mesh.visual.face_colors[:, :3] = face_colors

    scene.add_geometry(camera_mesh)

def apply_scene_alignment(scene_3d, extrinsics_matrices: np.ndarray):

    align_rotation = np.eye(4)

    align_rotation[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()

    initial_transformation = np.linalg.inv(extrinsics_matrices[0]) @ get_opengl_conversion_matrix() @ align_rotation

    scene_3d.apply_transform(initial_transformation)

    return scene_3d

def get_opengl_conversion_matrix() -> np.ndarray:

    matrix = np.identity(4)

    matrix[1, 1] = -1

    matrix[2, 2] = -1

    return matrix

def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int = None) -> np.ndarray:

    points = np.asarray(points)

    initial_shape = points.shape[:-1]

    dim = dim or points.shape[-1]

    transformation = transformation.swapaxes(-1, -2)

    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]

    return points[..., :dim].reshape(*initial_shape, dim)

def compute_camera_faces(cone_shape) -> np.ndarray:

    faces_list = []

    num_vertices_cone = len(cone_shape.vertices)

    for face in cone_shape.faces:

        if 0 in face:

            continue

        v1, v2, v3 = face

        v1_offset, v2_offset, v3_offset = face + num_vertices_cone

        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone

        faces_list.extend(

            [

                (v1, v2, v2_offset),

                (v1, v1_offset, v3),

                (v3_offset, v2, v3),

                (v1, v2, v2_offset_2),

                (v1, v1_offset_2, v3),

                (v3_offset_2, v2, v3),

            ]

        )

    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]

    return np.array(faces_list)
