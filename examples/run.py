import warp as wp
import numpy as np
from gen3d.warp.dataloading import get_ycbv_data, get_ycbv_num_images, get_ycb_mesh
import gen3d.warp.dataloading
from gen3d.warp.utils import (
    xyz_from_depth_image,
    pose_matrix_to_wp_transform,
    transform_points,
)
import gen3d.warp.viz
import gen3d
import fire
import scipy.stats
import glob

from inference import inference_step
from tqdm import tqdm
import os

from gen3d.utils.metrics import add_err, adds_err, compute_auc

from scipy.spatial.transform import Rotation as R
from collections import defaultdict
import pandas as pd
from gen3d.utils.fp_loader import YCBVTrackingResultLoader
import gc


def viz_observed_rgbd(observed_rgbd, fx, fy, cx, cy):
    o = observed_rgbd.numpy()
    xyz = xyz_from_depth_image(o[..., 3], fx, fy, cx, cy)
    rgb = o[..., :3]
    gen3d.rr_log_cloud(xyz, "scene/observed", colors=rgb)


def viz_model_prediction(vertices, colors, pose_estimate, gt_pose=None):
    gen3d.rr_log_cloud(vertices.numpy(), "model", colors=colors.numpy())
    gen3d.rr_log_cloud(
        transform_points(pose_estimate, vertices).numpy(),
        "scene/model",
        colors=colors.numpy(),
    )
    gen3d.warp.viz.rr_log_wp_transform(pose_estimate, "pose_estimate")
    if gt_pose is not None:
        gen3d.warp.viz.rr_log_wp_transform(gt_pose, "gt_pose")


def np_posquat_to_wp_transform(posquat):
    return wp.transform(posquat[:3], wp.quat(posquat[3:]))


def np_transform_to_np_posematrix(np_posquat):
    pos = np_posquat[:3]
    quat = np_posquat[3:]
    romatrix = R.from_quat(quat).as_matrix()
    return np.vstack(
        [np.hstack([romatrix, pos.reshape(3, -1)]), np.array([0, 0, 0, 1])]
    )


def run(scene=None, object=None, FRAME_RATE=1, rerun=False):
    gen3d.rr_init("run")
    ycb_dir = gen3d.get_root_path() / "assets/bop/ycbv/test"
    fp_loader = YCBVTrackingResultLoader(frame_rate=FRAME_RATE, split=ycb_dir.name)

    if scene is None:
        scenes = sorted(
            [int(os.path.split(x)[-1]) for x in glob.glob(os.path.join(ycb_dir, "*"))]
        )
    elif isinstance(scene, tuple):
        scenes = list(range(scene[0], scene[1] + 1))
    elif isinstance(scene, int):
        scenes = [scene]
    elif isinstance(scene, list):
        scenes = scene

    print(f"Scenes: {scenes}")

    num_poses = 2000
    num_vertices = 10000

    position_deltas = np.random.normal(0.0, 0.005, size=(num_poses, 3))
    quaternion_deltas = scipy.stats.vonmises_fisher(
        mu=np.array([0, 0, 0, 1]),
        kappa=2000.0,
    ).rvs(num_poses)

    include_identity = True
    if include_identity:
        position_deltas[0, :] = 0.0
        quaternion_deltas[0, :] = np.array([0, 0, 0, 1])

    pose_deltas = wp.array(
        np.hstack((position_deltas, quaternion_deltas)), dtype=wp.transform
    )

    pose_hypotheses = wp.empty(num_poses, dtype=wp.transform)
    pixel_coordinates = wp.zeros((num_poses, num_vertices), dtype=wp.vec2i)
    corresponding_rgbd_per_pose_and_vertex = wp.empty(
        (num_poses, num_vertices), dtype=wp.vec4
    )
    scores_per_pose_and_vertex = wp.empty((num_poses, num_vertices), dtype=float)
    scores_per_pose = wp.zeros(num_poses, dtype=float)

    for scene_id in scenes:
        print(f"==== Scene {scene_id} =====")
        num_images = get_ycbv_num_images(ycb_dir, scene_id)
        images_indices = range(1, num_images + 1, FRAME_RATE)
        all_data = get_ycbv_data(ycb_dir, scene_id, images_indices, fields=[])
        fx, fy, cx, cy = all_data[0]["camera_intrinsics"]
        YCB_NAMES = gen3d.warp.dataloading.YCB_MODEL_NAMES

        object_indices = (
            [object] if object is not None else range(len(all_data[0]["object_types"]))
        )
        print(f"Object indices: {object_indices}")

        def gt_wp_transform(all_data, T, object_id):
            return pose_matrix_to_wp_transform(
                np.linalg.inv(all_data[T]["camera_pose"])
                @ all_data[T]["object_poses"][object_id]
            )

        for object_id in object_indices:
            print(f"==== Object {object_id} =====")
            gen3d.rr_init("run")

            object_name = YCB_NAMES[all_data[0]["object_types"][object_id]]

            mesh = get_ycb_mesh(ycb_dir, all_data[0]["object_types"][object_id])

            T = 0
            initial_pose = gt_wp_transform(all_data, T, object_id)

            mask = all_data[0]["masks"][object_id]
            vertices = xyz_from_depth_image(
                all_data[T]["rgbd"][:, :, 3], fx, fy, cx, cy
            )[mask]
            colors = all_data[T]["rgbd"][:, :, :3][mask]

            # Sample num_vertices points from the mesh
            if len(vertices) < num_vertices:
                remaining = num_vertices - len(vertices)
                indices = np.random.choice(len(vertices), remaining)
                full_indices = np.concatenate([np.arange(len(vertices)), indices])
                vertices = vertices[full_indices]
                colors = colors[full_indices]
            else:
                indices = np.random.choice(len(vertices), num_vertices, replace=False)
                vertices = vertices[indices]
                colors = colors[indices]

            assert len(vertices) == num_vertices, (len(vertices), num_vertices)
            assert len(colors) == num_vertices

            vertices = transform_points(
                wp.transform_inverse(initial_pose),
                wp.array(vertices, dtype=wp.vec3),
            )
            colors = wp.array(colors, dtype=wp.vec3)

            gen3d.rr_set_time(0)

            print(
                f"Running on object_id {object_name} with {num_poses} poses and {num_vertices} vertices"
            )

            pose_estimate = wp.array(
                [initial_pose],
                dtype=wp.transform,
            )

            results = {}
            # for T in tqdm(range(0, len(all_data))):
            for T in tqdm(range(0, min(len(all_data), 5))):  # just do three frames for profiling
                image = wp.array(all_data[T]["rgbd"], dtype=wp.vec4)
                for _ in range(4):
                    inference_step(
                        pose_estimate,
                        pose_deltas,
                        image,
                        vertices,
                        colors,
                        fx,
                        fy,
                        cx,
                        cy,
                        pose_hypotheses,
                        pixel_coordinates,
                        corresponding_rgbd_per_pose_and_vertex,
                        scores_per_pose_and_vertex,
                        scores_per_pose,
                    )
                results[T] = np.array(pose_estimate.list()[0])

            # ALL_METRICS = {
            #     "ADD-S": adds_err,
            #     "ADD": add_err,
            # }

            # all_scores = {}
            # aggregated_auc = {}
            # fp_aggregated_auc = {}
            # precomputed_metrics = fp_loader.get_metrics(scene_id, object_id)
            # for metric_name, metric_fn in ALL_METRICS.items():
            #     all_scores[metric_name] = defaultdict(list)
            #     for T in tqdm(range(0, len(all_data))):
            #         value = float(
            #             metric_fn(
            #                 np_transform_to_np_posematrix(results[T]),
            #                 np_transform_to_np_posematrix(
            #                     np.array(gt_wp_transform(all_data, T, object_id))
            #                 ),
            #                 mesh.vertices,
            #             )
            #         )
            #         all_scores[metric_name][object_name].append(value)

            #     aggregated_auc[metric_name] = {}
            #     fp_aggregated_auc[metric_name] = {}
            #     for obj_name in all_scores[metric_name]:
            #         aggregated_auc[metric_name][obj_name] = float(
            #             compute_auc(all_scores[metric_name][obj_name])
            #         )

            #         fp_aggregated_auc[metric_name][object_name] = float(
            #             compute_auc(precomputed_metrics[metric_name][object_name])
            #         )

            # aggregated_auc_pd = pd.DataFrame(aggregated_auc)
            # fp_aggregated_auc_pd = pd.DataFrame(fp_aggregated_auc)
            # print(
            #     fp_aggregated_auc_pd.compare(
            #         aggregated_auc_pd, result_names=("FoundationPose", "Gen3D")
            #     )
            # )

        del vertices
        del colors
        del mesh
        del results
        del pose_estimate
        del all_data
        gc.collect()


if __name__ == "__main__":
    fire.Fire(run)