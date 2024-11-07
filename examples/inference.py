import warp as wp
import numpy as np


@wp.kernel
def apply_pose_deltas(
    pose_center: wp.array(dtype=wp.transform),
    pose_deltas: wp.array(dtype=wp.transform),
    pose_hypotheses: wp.array(dtype=wp.transform),
):
    """
    Set pose_hypotheses[idx] = pose_center[0] @ pose_deltas[idx].

    This kernel is used to produce a collection of poses nearby to pose_center[0].
    """
    pose_index = wp.tid()
    pose_hypotheses[pose_index] = wp.transform_multiply(
        pose_center[0], pose_deltas[pose_index]
    )


@wp.func
def pixel_likelihood(
    pixel_rgbd: wp.vec4,
    latent_color: wp.vec3,
    latent_depth: float,
    is_outlier: wp.bool,
):
    """
    Calculate pixel likelihood of observed RGBD pixel given
    the latent vertices color, depth, and outlier status.

    Args:
        pixel_rgbd: vec4f - [0,1]^3 x [near,far]) + {(-1, -1, -1, -1)}
        latent_color: vec3f - [0,1]^3
        latent_depth: float - [0, +inf)
        is_outlier: bool
    Output:
        log P(pixel_rgbd | latent_color, latent_depth, is_outlier)
    """
    color_match = (
        wp.max(
            wp.abs(wp.vec3(pixel_rgbd[0], pixel_rgbd[1], pixel_rgbd[2]) - latent_color)
        )
        < 0.05
    )
    depth_match = wp.abs(pixel_rgbd[3] - latent_depth) < 0.001

    if is_outlier:
        return wp.log(1.0 / 1.0) + wp.log(1.0 / 1000.0)
    else:
        return wp.log(float(color_match)) + wp.log(float(depth_match))


@wp.kernel
def score_pose_and_vertex(
    rgbd_image: wp.array(dtype=wp.vec4, ndim=2),
    vertices: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
    pose_hypotheses: wp.array(dtype=wp.transform),
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    pixel_coordinates: wp.array(dtype=wp.vec2i, ndim=2),
    corresponding_rgbd_per_pose_and_vertex: wp.array(dtype=wp.vec4, ndim=2),
    scores_per_pose_and_vertex: wp.array(dtype=float, ndim=2),
):
    """
    Kernel to calculate likelihood for all poses and all vertices.

    Args:
        rgbd_image: (H,W,4) - Observed RGBD image
        vertices: (N, 3) - 3D coordinates representing object shape
        colors: (N, 3) - RGB color associated with each of the vertices
        pose_hypotheses: (K,) - Pose hypotheses to consider
        fx: Focal length in x
        fy: Focal length in y
        cx: Principal point in x
        cy: Principal point in y
        pixel_coordinates: (K, N, 2) - Pixel coordinates applying a pose to the vertex and projecting to the image
        corresponding_rgbd_per_pose_and_vertex: (K, N, 4) - Observed RGBD values at pixel_coordinates
        scores_per_pose_and_vertex: (K, N) - Likelihood of each pose and vertex
    """
    # Each thread will process a pair of a pose hypothesis and a vertex.
    (pose_index, vertex_index) = wp.tid()

    # Transform the assigned vertex by the the assigned pose hypothesis.
    transformed_vertex = wp.transform_point(
        pose_hypotheses[pose_index], vertices[vertex_index]
    )

    # Project the transformed vertex to the image plane.
    pixel_raw = wp.vec2i(
        wp.int32(fy * transformed_vertex[1] / transformed_vertex[2] + cy),
        wp.int32(fx * transformed_vertex[0] / transformed_vertex[2] + cx),
    )

    # Check whether the projection is valid.
    # (1) Is it within the dimensions of the image.
    # (2) Is it in front of the camera.
    height = rgbd_image.shape[0]
    width = rgbd_image.shape[1]
    valid = (
        pixel_raw[0] >= 0
        and pixel_raw[0] < height
        and pixel_raw[1] >= 0
        and pixel_raw[1] < width
        and transformed_vertex[2] > 0
    )

    # Clip the pixel coordinates to the image dimensions to avoid out-of-bounds access.
    pixel = wp.vec2i(
        min(max(pixel_raw[0], 0), height - 1),
        min(max(pixel_raw[1], 0), width - 1),
    )

    # Get the observed RGBD pixel at the projected pixel coordinates.
    observed_rgbd_pixel = rgbd_image[pixel[0], pixel[1]]
    # If it's invalid, then set the pixel values to be invalid.
    if not valid:
        observed_rgbd_pixel = wp.vec4(-1.0, -1.0, -1.0, -1.0)

    # Now grid over the outlier status of the pixel and calculate the scores.
    scores = wp.vector(0.0, 0.0)

    sweep_over_is_outlier = wp.vector(wp.bool(True), wp.bool(False))
    for i in range(wp.static(2)):
        scores[i] = pixel_likelihood(
            observed_rgbd_pixel,
            colors[vertex_index],
            transformed_vertex[2],
            sweep_over_is_outlier[i],
        )

    # Record the pixel likelihood score.
    scores_per_pose_and_vertex[pose_index, vertex_index] = scores[
        int(wp.argmax(scores))
    ]

    #### All the following code is outputs that are used for debugging. ###

    # Save the pixel coordinates for the pose and vertex.
    pixel_coordinates[pose_index, vertex_index] = pixel_raw
    # This is for debugging
    corresponding_rgbd_per_pose_and_vertex[pose_index, vertex_index] = (
        observed_rgbd_pixel
    )


@wp.kernel
def accumulate_scores(
    scores_per_pose_and_vertex: wp.array(dtype=float, ndim=2),
    scores_per_pose: wp.array(dtype=float),
):
    """
    Accumulate the scores for each pose by summing over all vertices. (i.e. sum over the second dimension)

    Args:
        scores_per_pose_and_vertex: (K, N) - Likelihood of each pose and vertex pair
        scores_per_pose: (K,) - Likelihood of all vertices for each pose
    """
    pose_index = wp.tid()
    accumulated_score = float(0.0)
    for vertex_index in range(scores_per_pose_and_vertex.shape[1]):
        accumulated_score = (
            accumulated_score + scores_per_pose_and_vertex[pose_index, vertex_index]
        )
    scores_per_pose[pose_index] = accumulated_score


@wp.kernel
def select_best_pose(
    pose_hypotheses: wp.array(dtype=wp.transform),
    scores_per_pose: wp.array(dtype=float),
    pose_estimate: wp.array(dtype=wp.transform),
):
    """
    Select the pose hypothesis with the highest score.

    Args:
        pose_hypotheses: (K,) - Pose hypotheses to consider
        scores_per_pose: (K,) - Likelihood of each pose
        pose_estimate: (1,) - The best pose estimate
    """
    best_index = int(0)
    best_value = float(-np.inf)
    for i in range(pose_hypotheses.shape[0]):
        if scores_per_pose[i] > best_value:
            best_value = scores_per_pose[i]
            best_index = i
    pose_estimate[0] = pose_hypotheses[best_index]


def inference_step(
    pose_estimate: wp.array(dtype=wp.transform),
    pose_deltas: wp.array(dtype=wp.transform),
    rgbd_image: wp.array(dtype=wp.vec4, ndim=2),
    vertices: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    # These inputs are empty memory that will be filled by the kernels.
    pose_hypotheses: wp.array(dtype=wp.transform),
    pixel_coordinates: wp.array(dtype=wp.vec2i, ndim=2),
    corresponding_rgbd_per_pose_and_vertex: wp.array(dtype=wp.vec4, ndim=2),
    scores_per_pose_and_vertex: wp.array(dtype=float, ndim=2),
    scores_per_pose: wp.array(dtype=float),
):
    """
    Calls each of the kernels in the inference pipeline.

    Args:
        pose_estimate: (1,) - The current pose estimate
        pose_deltas: (K,) - Pose deltas to apply to the pose estimate
        rgbd_image: (H,W,4) - Observed RGBD image
        vertices: (N, 3) - 3D coordinates representing object shape
        colors: (N, 3) - RGB color associated with each of the vertices
        fx: float - Focal length in x
        fy: float - Focal length in y
        cx: float - Principal point in x
        cy: float - Principal point in y

        # These inputs are empty memory that will be filled by the kernels.

        pose_hypotheses: (K,) - Pose hypotheses to consider
        pixel_coordinates: (K, N, 2) - Pixel coordinates applying a pose to the vertex and projecting to the image
        corresponding_rgbd_per_pose_and_vertex: (K, N, 4) - Observed RGBD values at pixel_coordinates
        scores_per_pose_and_vertex: (K, N) - Likelihood of each pose and vertex
        scores_per_pose: (K,) - Likelihood of each pose
    """

    num_poses = pose_deltas.shape[0]
    num_points = vertices.shape[0]

    wp.launch(
        kernel=apply_pose_deltas,
        dim=num_poses,
        inputs=[pose_estimate, pose_deltas, pose_hypotheses],
    )
    wp.launch(
        kernel=score_pose_and_vertex,
        dim=(num_poses, num_points),
        inputs=[
            rgbd_image,
            vertices,
            colors,
            pose_hypotheses,
            fx,
            fy,
            cx,
            cy,
            pixel_coordinates,
            corresponding_rgbd_per_pose_and_vertex,
            scores_per_pose_and_vertex,
        ],
    )
    wp.launch(
        kernel=accumulate_scores,
        dim=(num_poses),
        inputs=[scores_per_pose_and_vertex, scores_per_pose],
    )
    wp.launch(
        kernel=select_best_pose,
        dim=(1),
        inputs=[pose_hypotheses, scores_per_pose, pose_estimate],
    )