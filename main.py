import torch 
import torch.nn.functional as F
import torchvision
import tyro
import viser
import trimesh
import viser.transforms as tf
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import os 

from tqdm import tqdm
from search import Search
from optimize import Optimize
from utils import sample_texture, grid_show, tensor_show, pointify_tensor

def randomize_voxels(full_grid, texture, padding=0):
	full_grid_tensor = torch.from_numpy(full_grid.matrix).unsqueeze(-1).expand(-1,-1,-1,3).float()
	
	num_samples = full_grid_tensor.shape[0] * full_grid_tensor.shape[1] * full_grid_tensor.shape[2]
	num_pixels = texture.shape[0] * texture.shape[1]
	p = torch.ones(num_pixels)/num_pixels
	index = torch.multinomial(input=p, num_samples=num_samples, replacement=True)
	sample_colors = texture.view((num_pixels, -1))[index].reshape(full_grid_tensor.shape[0], full_grid_tensor.shape[1], full_grid_tensor.shape[2], 3)
	full_grid_tensor = full_grid_tensor * sample_colors
	return full_grid_tensor


def sample_voxel(full_grid_mask: torch.Tensor, batch_size: int = 16, neighborhood_dim: int=8) -> torch.Tensor:
	"""
	full_grid:  torch.Tensor (d, h, w)
	batch size: number of voxels sampling for neighborhoods

	return: torch.Tensor (batch_size, 3)
	"""
	#select random voxel 
	d, h, w = full_grid_mask.shape
	padding = neighborhood_dim//2
	
	h_index = torch.randint(padding, h-padding, size=(batch_size,1))
	w_index = torch.randint(padding, w-padding, size=(batch_size,1))
	if d == 1:
		d_index = torch.zeros_like(h_index)
	else:
		d_index = torch.randint(padding, d-padding, size=(batch_size,1))
	index = torch.hstack([d_index, h_index, w_index])
	true_index = index[full_grid_mask[index.T[0],index.T[1],index.T[2]]]
	if len(true_index)== 0: 
		return sample_voxel(full_grid_mask, batch_size,neighborhood_dim)
	else:
		return true_index

	
def sample_neighborhood(full_grid_tensor: torch.Tensor, index: torch.Tensor, neighborhood_dim: int = 8) -> torch.Tensor:
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
		if full_grid_tensor.shape[0] > 1:
			xz_grid = full_grid_tensor[z_start_indices[i]:z_end_indices[i], index[:, 1][0], y_start_indices[i]:y_end_indices[i], :].unsqueeze(0)
			yz_grid = full_grid_tensor[z_start_indices[i]:z_end_indices[i], x_start_indices[i]:x_end_indices[i], index[:, 2][0], :].unsqueeze(0)
			neighborhood.append(torch.vstack([xy_grid, xz_grid, yz_grid]).unsqueeze(0))
		else:
			neighborhood.append(xy_grid.unsqueeze(0))
	try:
		neighborhood = torch.vstack(neighborhood)
	except:
		import pdb; pdb.set_trace()
	
	return neighborhood


def custom_interpolate(x, scale_factor, mode=None): 
	# TODO fix the mode of interpolation to bilinear and bicubic
	# F.interpolate expects (B C ...)
	if len(x.shape) == 3:
		# 2D texture case (H, W, C) 
		x = F.interpolate(x.permute(2,0,1).unsqueeze(0), scale_factor=scale_factor)
		return x[0].permute(1,2,0)
	elif len(x.shape) == 4 and x.shape[0] == 1:
		# 2D "solid" case (D, H, W, C) with depth = 1 which should NOT be downsampled 
		x = F.interpolate(x.permute(3,0,1,2).unsqueeze(0), scale_factor=(1, scale_factor, scale_factor))
		return x[0].permute(1,2,3,0)
	elif len(x.shape) == 4 and x.shape[0] > 1:
		# 3D solid case (D, H, W, C) 
		x = F.interpolate(x.permute(3,0,1,2).unsqueeze(0), scale_factor=scale_factor)
		return x[0].permute(1,2,3,0)
	raise Exception("Uh oh, error in interpolation")


def main(texture_file: str = 'zebra.png', 
		 object_file: str = 'cow.obj',
		 texture_dir: str = 'textures', 
		 object_dir: str = "objs", 
		 pitch: float = 0.1,
		 num_iters: int = 4,
		 show: bool = True, 
		 batch_size: int = 32, 
		 show_3d: bool = True,
		 test_2d: bool = False,
		 neighborhood_dim: int = 8,
		 r: float = 0.8,
		 resolutions: list[float] = [1],
		 display_freq: int = 4,
		 use_hist: bool = True,
		 experiment_name: str = None,
		 device: str = 'cpu',
		 ):
	assert num_iters >= 4, "Iterate more. 4 is too few"
	
	if experiment_name is None:
		now = datetime.now()
		experiment_name = now.strftime("%m-%d-%Y_%H-%M-%S")
	# Load and sample texture
	texture = torchvision.io.read_image(texture_dir + '/' + texture_file).permute(1, 2, 0).float() / 255.0
	
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
	
		full_grid_tensor = randomize_voxels(full_grid, texture, padding=neighborhood_dim)
		mask = full_grid.matrix
		# convert mask to tensor
		mask = torch.from_numpy(mask).bool()
	else:
		full_grid_tensor = sample_texture(
			texture, 
			(1, neighborhood_dim**2 + neighborhood_dim, neighborhood_dim**2 + neighborhood_dim, 3))
		mask = torch.ones_like(full_grid_tensor[:, :, :, 0]).bool()
	
	downsampled_full_grid = custom_interpolate(full_grid_tensor, scale_factor=resolutions[0])
	for r in range(len(resolutions)):
		scale = resolutions[r]
		print(f"Commencing optimization at resolution {scale}")
		downsampled_texture = custom_interpolate(texture, scale_factor=scale)
		downsampled_mask = custom_interpolate(mask.float().unsqueeze(-1), scale_factor=scale).bool().squeeze(-1)
		if any(torch.tensor(downsampled_mask.shape) <= neighborhood_dim):
			print(f"Skipping resolution {scale} (too downsampled)")
			continue 
		tensor_show(downsampled_full_grid[:, neighborhood_dim:-neighborhood_dim, neighborhood_dim:-neighborhood_dim, :], show=show)
		if show:
			plt.imshow(downsampled_texture)
			plt.show()

		optimize = Optimize(downsampled_texture, r=r, use_hist=use_hist)
		search = Search(downsampled_texture, neighborhood_dim=neighborhood_dim, index=r, experiment_name=experiment_name)
		for i in tqdm(range(int(num_iters * scale))):
			index = sample_voxel(downsampled_mask, batch_size=batch_size, neighborhood_dim=neighborhood_dim)
			neighborhood = sample_neighborhood(downsampled_full_grid, index, neighborhood_dim=neighborhood_dim)
			texel_match = search.find(neighborhood)
			
			new_value = optimize(exemplar=texel_match, solid=neighborhood)
			downsampled_full_grid[index.T[0], index.T[1], index.T[2]] = new_value

			grid_show(texels=texel_match, voxels=neighborhood, show=show and i%(num_iters//display_freq) == 0)
			tensor_show(downsampled_full_grid, show=show and i%(num_iters//display_freq) == 0)	
		if r + 1 < len(resolutions):
			print(f"Upsampling optimized tensor to resolution {resolutions[r+1]}")
			downsampled_full_grid = custom_interpolate(
				downsampled_full_grid, 
				scale_factor=int(resolutions[r+1]/resolutions[r]),
				mode='bicubic')

	if test_2d:
		downsampled_full_grid = downsampled_full_grid[:, neighborhood_dim:-neighborhood_dim, neighborhood_dim:-neighborhood_dim, :]
	
	tensor_show(downsampled_full_grid, show=True)
	if test_2d:
		os.makedirs("outputs/", exist_ok=True)
		hist = "" if use_hist else "no"
		plt.imsave(f'outputs/zebra_{hist}_hist_resolutions_{"_".join(map(str,  resolutions))}_{num_iters}_iters.png', downsampled_full_grid[0].numpy())
	else: 
		colors = pointify_tensor(full_grid_tensor, mask=mask)
	
	search.remove_cache()
 

	# display mesh
	if not test_2d and show_3d:
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

		server.add_point_cloud(
			name="/texture_voxels",
			points=full_grid.points,
			position=(0.0, 0.0, 0.0),
			colors=colors,
			wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
			point_size=0.1,
		)

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