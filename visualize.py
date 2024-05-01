import torch
import numpy as np
import time
import viser
import viser.transforms as tf
import tyro

from utils import pointify_tensor, generate_indices

def visualize_grid(full_grid, colors, pitch):
    server = viser.ViserServer()
    full_grid_min_x, full_grid_min_y, full_grid_min_z = full_grid.bounds[0]
    full_grid_max_x, full_grid_max_y, full_grid_max_z = full_grid.bounds[1]
    full_grid_min_x, full_grid_min_y, full_grid_min_z = float(full_grid_min_x), float(full_grid_min_y), float(full_grid_min_z)
    full_grid_max_x, full_grid_max_y, full_grid_max_z = float(full_grid_max_x), float(full_grid_max_y), float(full_grid_max_z)

    x_slice = server.add_gui_slider(
        "x_slice",
        min=full_grid_min_x,
        max=full_grid_max_x,
        step=pitch,
        initial_value=full_grid_max_x,
        disabled=False,
    )

    y_slice = server.add_gui_slider(
        "y_slice",
        min=full_grid_min_y,
        max=full_grid_max_y,
        step=pitch,
        initial_value=full_grid_max_y,
        disabled=False,
    )

    z_slice = server.add_gui_slider(
        "z_slice",
        min=full_grid_min_z,
        max=full_grid_max_z,
        step=pitch,
        initial_value=full_grid_max_z,
        disabled=False,
    )

    def show_pointcloud() -> None:
        curr_x_max = float(x_slice.value)
        curr_y_max = float(y_slice.value)
        curr_z_max = float(z_slice.value)

        visibility_mask = np.logical_and(full_grid.points[:, 0] < curr_x_max, np.logical_and(full_grid.points[:, 1] < curr_y_max, full_grid.points[:, 2] < curr_z_max))
        server.add_point_cloud(
            name="/texture_voxels",
            points=full_grid.points[visibility_mask],
            position=(0.0, 0.0, 0.0),
            colors=colors[visibility_mask],
            wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
            point_size=pitch/2,
        )

    x_slice.on_update(lambda _: show_pointcloud())
    y_slice.on_update(lambda _: show_pointcloud())
    z_slice.on_update(lambda _: show_pointcloud())

    while True:
        time.sleep(10.0)

def visualize_tensor(full_grid_tensor, full_grid_mask, pitch):
    colors = pointify_tensor(full_grid_tensor, full_grid_mask)
    
    full_grid_points = generate_indices(full_grid_mask, batch_size=-1, shuffle=False).float() * pitch # (N, 3)
    full_grid_min = torch.min(full_grid_points, dim=0).values
    full_grid_max = torch.max(full_grid_points, dim=0).values
    
    full_grid_points = full_grid_points.numpy()
    full_grid_min = full_grid_min.numpy()
    full_grid_max = full_grid_max.numpy()
    
    full_grid_min_x, full_grid_min_y, full_grid_min_z = float(full_grid_min[0]), float(full_grid_min[1]), float(full_grid_min[2])
    full_grid_max_x, full_grid_max_y, full_grid_max_z = float(full_grid_max[0]), float(full_grid_max[1]), float(full_grid_max[2])
    
    server = viser.ViserServer()
    x_slice = server.add_gui_slider(
        "x_slice",
        min=full_grid_min_x,
        max=full_grid_max_x,
        step=pitch,
        initial_value=full_grid_max_x,
        disabled=False,
    )

    y_slice = server.add_gui_slider(
        "y_slice",
        min=full_grid_min_y,
        max=full_grid_max_y,
        step=pitch,
        initial_value=full_grid_max_y,
        disabled=False,
    )

    z_slice = server.add_gui_slider(
        "z_slice",
        min=full_grid_min_z,
        max=full_grid_max_z,
        step=pitch,
        initial_value=full_grid_max_z,
        disabled=False,
    )

    def show_pointcloud() -> None:
        curr_x_max = float(x_slice.value)
        curr_y_max = float(y_slice.value)
        curr_z_max = float(z_slice.value)

        visibility_mask = np.logical_and(full_grid_points[:, 0] < curr_x_max, np.logical_and(full_grid_points[:, 1] < curr_y_max, full_grid_points[:, 2] < curr_z_max))
        server.add_point_cloud(
            name="/texture_voxels",
            points=full_grid_points[visibility_mask],
            position=(0.0, 0.0, 0.0),
            colors=colors[visibility_mask],
            wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
            point_size=pitch/2,
        )

    x_slice.on_update(lambda _: show_pointcloud())
    y_slice.on_update(lambda _: show_pointcloud())
    z_slice.on_update(lambda _: show_pointcloud())

    while True:
        time.sleep(10.0)

def vis_file(
    tensor_file: str, 
    pitch=0.01,
):
    full_grid_tensor = torch.load(tensor_file)
    full_grid_mask = torch.nonzero(full_grid_tensor)
    visualize_tensor(full_grid_tensor, full_grid_mask, pitch)

if __name__ == "__main__":
    full_grid_tensor = torch.load("./outputs/zebra_cow/voxel_grids/zebra_cow_final.pt")
    full_grid_mask = torch.max(full_grid_tensor, dim=3).values.bool()
    
    visualize_tensor(full_grid_tensor, full_grid_mask, 0.03)
    tyro.cli(vis_file)

