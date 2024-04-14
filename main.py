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

from search import Search
from optimize import Optimize
from utils import sample_texture, grid_show

def randomize_voxels(full_grid, texture):
	full_grid_tensor = torch.from_numpy(full_grid.matrix).unsqueeze(-1).expand(-1,-1,-1,3).float()
	
	num_samples = full_grid_tensor.shape[0] * full_grid_tensor.shape[1] * full_grid_tensor.shape[2]
	num_pixels = texture.shape[0] * texture.shape[1]
	p = torch.ones(num_pixels)/num_pixels
	index = torch.multinomial(input=p, num_samples=num_samples, replacement=True)
	sample_colors = texture.view((num_pixels, -1))[index].reshape(full_grid_tensor.shape[0], full_grid_tensor.shape[1], full_grid_tensor.shape[2], 3)
	full_grid_tensor = full_grid_tensor * sample_colors
	return full_grid_tensor

def sample_voxel(full_grid_mask: torch.Tensor, batch_size: int = 16, neighborhood_dim: int=8) -> torch.Tensor:
	if full_grid_mask.shape[0] == 1:
		return sample_voxel_2d(full_grid_mask, batch_size, neighborhood_dim)
	else: 
		return sample_voxel_3d(full_grid_mask, batch_size, neighborhood_dim)
	
def sample_voxel_2d(full_grid_mask: torch.Tensor, batch_size: int = 16, neighborhood_dim: int=8) -> torch.Tensor:
	"""
	full_grid:  torch.Tensor (d, h, w)
	batch size: number of voxels sampling for neighborhoods

	return: torch.Tensor (batch_size, 3)
	"""
	#select random voxel (16 times)
	_, h, w = full_grid_mask.shape
	padding = neighborhood_dim//2
	h_index = torch.randint(padding, h-padding, size=(batch_size,1))
	w_index = torch.randint(padding, w-padding, size=(batch_size,1))
	index = torch.hstack([h_index, w_index])
	return index[full_grid_mask[[0, index.T[0],index.T[1]]]]

def sample_voxel_3d(full_grid_mask: torch.Tensor, batch_size: int = 16, neighborhood_dim: int=8) -> torch.Tensor:
	"""
	full_grid:  torch.Tensor (d, h, w)
	batch size: number of voxels sampling for neighborhoods

	return: torch.Tensor (batch_size, 3)
	"""
	#select random voxel 
	d, h, w = full_grid_mask.shape
	padding = neighborhood_dim//2
	d_index = torch.randint(padding, d-padding, size=(batch_size,1))
	h_index = torch.randint(padding, h-padding, size=(batch_size,1))
	w_index = torch.randint(padding, w-padding, size=(batch_size,1))
	index = torch.hstack([d_index, h_index, w_index])
	return index[full_grid_mask[index.T[0],index.T[1],index.T[2]]]

def set_neighborhood(new_neighborhood: torch.Tensor, full_grid_tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
	"""
	new_neighborhood: torch.Tensor (batch_size, 3, neighborhood_dim, neighborhood_dim, 3)
	full_grid_tensor:  torch.Tensor (z, x, y, 3)
	index:  torch.Tensor (batch_size, 3)
	neighborhood_dim: (int) dimentions of neighborhood (neighborhood_dim * neighborhood_dim)

	return: torch.Tensor (batch_size, 3, neighborhood_dim, neighborhood_dim, 3)
	"""
	if index.shape[1] == 3:
		#find neighborhood of voxel
		neighborhood_dim = new_neighborhood.shape[2]
		x_start_indices = index[:, 1] - neighborhood_dim // 2
		x_end_indices = index[:, 1] + neighborhood_dim // 2
		y_start_indices = index[:, 2] - neighborhood_dim // 2
		y_end_indices = index[:, 2] + neighborhood_dim // 2
		z_start_indices = index[:, 0] - neighborhood_dim // 2
		z_end_indices = index[:, 0] + neighborhood_dim // 2
		
		for i in range(index.shape[0]):
			full_grid_tensor[z_start_indices[i]:z_end_indices[i], x_start_indices[i]:x_end_indices[i], y_start_indices[i]:y_end_indices[i], :] = new_neighborhood[i]
	else: 
		neighborhood_dim = new_neighborhood.shape[2]
		x_start_indices = index[:, 0] - neighborhood_dim // 2
		x_end_indices = index[:, 0] + neighborhood_dim // 2
		y_start_indices = index[:, 1] - neighborhood_dim // 2
		y_end_indices = index[:, 1] + neighborhood_dim // 2
		
		for i in range(index.shape[0]):
			full_grid_tensor[0, x_start_indices[i]:x_end_indices[i], y_start_indices[i]:y_end_indices[i], :] = new_neighborhood[i]
	
	return full_grid_tensor

def sample_neighborhood(full_grid_tensor: torch.Tensor, index: torch.Tensor, neighborhood_dim: int = 8) -> torch.Tensor:
	if full_grid_tensor.shape[0] == 1:
		return sample_neighborhood_2d(full_grid_tensor, index, neighborhood_dim)
	else:
		return sample_neighborhood_3d(full_grid_tensor, index, neighborhood_dim)

	
def sample_neighborhood_3d(full_grid_tensor: torch.Tensor, index: torch.Tensor, neighborhood_dim: int = 8, show: bool = False) -> torch.Tensor:
	"""
	full_grid_tensor:  torch.Tensor (z, x, y, 3)
	index:  torch.Tensor (batch_size, 3)
	neighborhood_dim: (int) dimentions of neighborhood (neighborhood_dim * neighborhood_dim)

	return: torch.Tensor (batch_size, 3, neighborhood_dim, neighborhood_dim, 3)
	"""
	assert(neighborhood_dim % 2 == 0)
	#find neighborhood of voxel

	x_start_indices = index[:, 1] - neighborhood_dim // 2
	x_end_indices = index[:, 1] + neighborhood_dim // 2
	y_start_indices = index[:, 2] - neighborhood_dim // 2
	y_end_indices = index[:, 2] + neighborhood_dim // 2
	z_start_indices = index[:, 0] - neighborhood_dim // 2
	z_end_indices = index[:, 0] + neighborhood_dim // 2
	
	neighborhood = []
	for i in range(index.shape[0]):
		
		xy_grid = full_grid_tensor[index[:, 0][0], x_start_indices[i]:x_end_indices[i], y_start_indices[i]:y_end_indices[i], :].unsqueeze(0)
		xz_grid = full_grid_tensor[z_start_indices[i]:z_end_indices[i], index[:, 1][0], y_start_indices[i]:y_end_indices[i], :].unsqueeze(0)
		yz_grid = full_grid_tensor[z_start_indices[i]:z_end_indices[i], x_start_indices[i]:x_end_indices[i], index[:, 2][0], :].unsqueeze(0)
		neighborhood.append(torch.vstack([xy_grid, xz_grid, yz_grid]).unsqueeze(0))
	
	neighborhood = torch.vstack(neighborhood)
	
	if show:
		plt.imshow(neighborhood[0,0,:,:,:])
		plt.show()
	return neighborhood

def sample_neighborhood_2d(full_grid_tensor: torch.Tensor, index: torch.Tensor, neighborhood_dim: int = 8) -> torch.Tensor:
	"""
	full_grid_tensor:  torch.Tensor (1, x, y, 3)
	index:  torch.Tensor (batch_size, 2)
	neighborhood_dim: (int) dimentions of neighborhood (neighborhood_dim * neighborhood_dim)
	"""
	assert(neighborhood_dim % 2 == 0)
	#find neighborhood of voxel

	x_start_indices = index[:, 0] - neighborhood_dim // 2
	x_end_indices = index[:, 0] + neighborhood_dim // 2
	y_start_indices = index[:, 1] - neighborhood_dim // 2
	y_end_indices = index[:, 1] + neighborhood_dim // 2
	
	neighborhood = []
	for i in range(index.shape[0]):
		neighborhood.append(full_grid_tensor[:, x_start_indices[i]:x_end_indices[i], y_start_indices[i]:y_end_indices[i], :].unsqueeze(0))
	
	neighborhood = torch.vstack(neighborhood)
	return neighborhood


def main(texture_file: str = 'tomatoes.png', 
		 object_file: str = 'cow.obj', 
		 texture_dir: str = 'textures', 
		 object_dir: str = "objs", 
		 pitch: float = 0.1,
		 num_iters: int = 2,
		 show: bool = True, 
		 test_2d: bool = False,
		 device: str = 'cpu'):
	
	# Load and sample texture
	texture = torchvision.io.read_image(texture_dir + '/' + texture_file).permute(1, 2, 0).float()/255.0
	
	if not test_2d:
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
	
		full_grid_tensor = randomize_voxels(full_grid, texture)
		mask = full_grid.matrix
	else:
		full_grid_tensor = sample_texture(texture, (1, 32, 32, 3))
		mask = torch.ones_like(full_grid_tensor[:, :, :, 0]).bool()
	
	search = Search(texture)
	if show:
		plt.imshow(full_grid_tensor[0,:,:,:].numpy())
		plt.show()
	
	for i in range(num_iters):
		index = sample_voxel(mask)
		neighborhood = sample_neighborhood(full_grid_tensor, index, neighborhood_dim=8)
		texel_match = search.find(neighborhood)
		grid_show(texels=texel_match, voxels=neighborhood)
		optimize = Optimize()
		new_neighborhood = optimize(exemplar=texel_match, solid=neighborhood)
		full_grid_tensor = set_neighborhood(new_neighborhood, full_grid_tensor, index)
		if show:
			plt.imshow(full_grid_tensor[0,:,:,:].numpy())
			plt.show()
		




	# display mesh
	if False and show:
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