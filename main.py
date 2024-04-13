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
import pdb


def sample_voxel(full_grid: torch.Tensor, batch_size: int = 16) -> torch.Tensor:
	"""
	full_grid:  torch.Tensor (z, x, y, 3)
	batch size: number of voxels sampling for neighborhoods

	return: torch.Tensor (batch_size, 3)
	"""
	#select random voxel (16 times)
	p = torch.ones(full_grid.shape[0] * full_grid.shape[1] * full_grid.shape[2])/(full_grid.shape[0] * full_grid.shape[1] * full_grid.shape[2])
	index = torch.multinomial(input=p, num_samples=batch_size, replacement=True)
	index = torch.stack([index // (full_grid.shape[1] * full_grid.shape[2]), \
					 	(index % (full_grid.shape[1] * full_grid.shape[2])) // full_grid.shape[2], \
						(index % (full_grid.shape[1] * full_grid.shape[2])) % full_grid.shape[2]], dim=1)
	return index



def sample_neighborhood(full_grid_tensor: torch.Tensor, index: torch.Tensor, neighborhood_dim: int = 8) -> torch.Tensor:
	"""
	full_grid_tensor:  torch.Tensor (z, x, y, 3)
	index:  torch.Tensor (batch_size, 3)
	neighborhood_dim: (int) dimentions of neighborhood (neighborhood_dim * neighborhood_dim)

	return: torch.Tensor (batch_size, 3, neighborhood_dim, neighborhood_dim, 3)
	"""
	#find neighborhood of voxel
	pdb.set_trace()
	indices_expanded = index[:, None, :].expand(-1, neighborhood_dim, -1)
	offsets = torch.arange(-neighborhood_dim // 2, neighborhood_dim // 2).unsqueeze(0).unsqueeze(-1)


def main(texture_file: str = 'tomatoes.png', 
		 object_file: str = "cube.obj", 
		 texture_dir: str = 'textures', 
		 object_dir: str = "objs", 
		 pitch: float = 0.01,
		 show: bool = True, 
		 device: str = 'cpu'):
	
	# Load and sample texture
	texture = torchvision.io.read_image(texture_dir + '/' + texture_file).permute(1, 2, 0).float()/255.0
	num_samples = texture.shape[0] * texture.shape[1] 
	p = torch.ones(num_samples)/num_samples # Initialize a uniform distribution over samples
	index = torch.multinomial(input=p, num_samples=num_samples, replacement=True) #samples from uniform distribution
	init = texture.view((num_samples, -1))[index].reshape(texture.shape) # gets colors of sampled pixels from texture image + shapes into texture shape

	# load mesh
	mesh = trimesh.load(object_dir + '/' + object_file)
	vertices = mesh.vertices
	faces = mesh.faces
	volume = mesh.volume
	corners = mesh.bounds
	print(f"Loaded mesh with {vertices.shape} vertices, {faces.shape} faces, and volume {volume}")

	# voxelize mesh
	start = time.time()
	full_grid = trimesh.voxel.creation.voxelize(mesh, pitch=pitch).fill() #number of voxels in voxel grid (depth, length, width)
	end = time.time()
	print(f"Voxelized mesh with shape {full_grid.shape} in {end - start:.2f} seconds")
	full_grid_tensor = torch.from_numpy(full_grid.matrix).unsqueeze(-1).expand(-1,-1,-1,3).float()
	
	num_samples = full_grid_tensor.shape[0] * full_grid_tensor.shape[1] * full_grid_tensor.shape[2]
	num_pixels = texture.shape[0] * texture.shape[1]
	p = torch.ones(num_pixels)/num_pixels
	index = torch.multinomial(input=p, num_samples=num_samples, replacement=True)
	sample_colors = texture.view((num_pixels, -1))[index].reshape(full_grid_tensor.shape[0], full_grid_tensor.shape[1], full_grid_tensor.shape[2], 3)
	full_grid_tensor = full_grid_tensor * sample_colors
	
	sample_neighborhood(full_grid_tensor, sample_voxel(full_grid_tensor), neighborhood_dim=8)

	# display mesh
	if show:
		# plt.imshow( init )
		# plt.show()
	
		server = viser.ViserServer()
		
		# server.add_mesh_simple(
		# 	name="/mesh", 
		# 	vertices=vertices,
		# 	faces=faces,
		# 	wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
		# 	position=(0.0, 0.0, 0.0),
		# )

		# display voxels in viser and sample colors from texture
		server.add_point_cloud(
			name="/full_grid",
			points=full_grid_tensor[:, :, :, 0].nonzero().numpy() * pitch,
			position=(0.0, 0.0, 0.0),
			colors=(255, 0 , 0),
			wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
			point_size=0.01,
		)

		# server.add_point_cloud(
		# 	name="/sample_voxels1",
		# 	points=sample_voxels1.points,
		# 	position=(-2.0,-1.0,-1.0),
		# 	colors=sample_colors1,
		# 	# wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
		# 	point_size=0.01,
		# )

		# server.add_point_cloud(
		# 	name="/sample_voxels2",
		# 	points=sample_voxels2.points,
		# 	position=(-2.0,-0.5,-1.0),
		# 	colors=sample_colors2,
		# 	# wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
		# 	point_size=0.01,
		# )

		# server.add_point_cloud(
		# 	name="/mesh_pointcloud",
		# 	points=mesh_pointcloud[::100].numpy(),
		# 	position=(0.0, 0.0, 0.0),
		# 	colors=(255, 0, 0),
		# 	wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
		# 	point_size=0.5,
		# )

		while True:
			time.sleep(10.0)
	

if __name__ == '__main__':
	tyro.cli(main)