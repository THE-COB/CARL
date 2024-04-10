import torch 
import torchvision
import matplotlib.pyplot as plt
import tyro
from pathlib import Path
import viser
import trimesh
import viser.transforms as tf
import numpy as np
import time

def sample_voxels(sample_point, radius, mesh, texture, pitch=0.05):
	# Sample voxel mesh
	sample_grid = trimesh.voxel.creation.local_voxelize(mesh=mesh, point=sample_point, pitch=pitch, radius=radius).apply_scale(pitch)
	num_samples = sample_grid.points.shape[0]
	num_pixels = texture.shape[0] * texture.shape[1]
	p = torch.ones(num_pixels)/num_pixels
	index = torch.multinomial(input=p, num_samples=num_samples, replacement=True)
	sample_colors = texture.view((num_pixels, -1))[index].reshape(sample_grid.points.shape[0], 3).numpy()
	return sample_grid, sample_colors

def main(texture_file: str = 'tomatoes.png', 
		 object_file: str = "cube.obj", 
		 texture_dir: str = 'textures', 
		 object_dir: str = "objs", 
		 show: bool = False, 
		 device: str = 'cpu'):
	
	# Load and sample texture
	texture = torchvision.io.read_image(texture_dir + '/' + texture_file).permute(1, 2, 0)
	num_samples = texture.shape[0] * texture.shape[1] 
	p = torch.ones(num_samples)/num_samples
	index = torch.multinomial(input=p, num_samples=num_samples, replacement=True)
	init = texture.view((num_samples, -1))[index].reshape(texture.shape)

	# load mesh
	mesh = trimesh.load(object_dir + '/' + object_file)
	vertices = mesh.vertices
	faces = mesh.faces
	print(f"Loaded mesh with {vertices.shape} vertices, {faces.shape} faces")

	# voxelize mesh
	full_grid = trimesh.voxel.creation.voxelize(mesh, pitch=0.05)
	print(f"Voxelized mesh with shape {full_grid.shape}")
	num_samples = full_grid.points.shape[0]
	num_pixels = texture.shape[0] * texture.shape[1]
	p = torch.ones(num_pixels)/num_pixels
	index = torch.multinomial(input=p, num_samples=num_samples, replacement=True)
	full_colors = texture.view((num_pixels, -1))[index].reshape(full_grid.points.shape[0], 3)

	sample_voxels1, sample_colors1 = sample_voxels(np.array([0,0,0]), 100, mesh, texture, pitch=0.05)
	sample_voxels2, sample_colors2 = sample_voxels(np.array([-2.0, -1.0, -1.0]), 100, mesh, texture, pitch=0.05)

	# display mesh
	if show:
		# plt.imshow( init )
		# plt.show()
	
		server = viser.ViserServer()
		
		server.add_mesh_simple(
			name="/mesh",
			vertices=vertices,
			faces=faces,
			wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
			position=(0.0, 0.0, 0.0),
		)

		# display voxels in viser and sample colors from texture
		server.add_point_cloud(
			name="/full_grid",
			points=full_grid.points[::2],
			position=(0.0, 0.0, 0.0),
			colors=(0, 0, 255),
			wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
			point_size=0.01,
		)

		server.add_point_cloud(
			name="/sample_voxels1",
			points=sample_voxels1.points[::2],
			position=(0.0, 0.0, 0.0),
			colors=(255, 0, 0),
			# wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
			point_size=0.01,
		)

		server.add_point_cloud(
			name="/sample_voxels2",
			points=sample_voxels2.points[::2],
			position=(-2.0, -1.0, -1.0),
			colors=(0, 255, 0),
			# wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
			point_size=0.01,
		)
		
		while True:
			time.sleep(10.0)
	

if __name__ == '__main__':
	tyro.cli(main)