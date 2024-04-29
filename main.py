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

from einops import rearrange
from tqdm import tqdm
from search import Search
from optimize import Optimize
from utils import sample_texture, grid_show, tensor_show, pointify_tensor

def randomize_voxels(full_grid, texture, padding=0, device="cpu"):
	full_grid_tensor = torch.from_numpy(full_grid.matrix).unsqueeze(-1).expand(-1,-1,-1,3).float().to(device)
	
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
	device = full_grid_mask.device
	
	h_index = torch.randint(0, h, size=(batch_size,1), device=device)
	w_index = torch.randint(0, w, size=(batch_size,1), device=device)
	if d == 1:
		d_index = torch.zeros_like(h_index, device=device)
	else:
		d_index = torch.randint(0, d, size=(batch_size,1), device=device)
	index = torch.hstack([d_index, h_index, w_index])
	true_index = index[full_grid_mask[index.T[0],index.T[1],index.T[2]]]
	if len(true_index)== 0: 
		return sample_voxel(full_grid_mask, batch_size,neighborhood_dim)
	else:
		return true_index

	
def sample_neighborhood(full_grid_tensor_padded: torch.Tensor, index: torch.Tensor, neighborhood_dim: int = 8) -> torch.Tensor:
	"""
	full_grid_tensor:  torch.Tensor (z, x, y, 3)
	index:  torch.Tensor (batch_size, 3)
	neighborhood_dim: (int) dimentions of neighborhood (neighborhood_dim * neighborhood_dim)

	return: torch.Tensor (batch_size, 3, neighborhood_dim, neighborhood_dim, 3)
	"""
	assert(neighborhood_dim % 2 == 0)
	pad_adjustment = neighborhood_dim // 2
	
	#find neighborhood of voxel

	x_start_indices = index[:, 1] - neighborhood_dim // 2 + pad_adjustment
	x_end_indices = index[:, 1] + neighborhood_dim // 2 + pad_adjustment
	y_start_indices = index[:, 2] - neighborhood_dim // 2 + pad_adjustment
	y_end_indices = index[:, 2] + neighborhood_dim // 2 + pad_adjustment
	z_start_indices = index[:, 0] - neighborhood_dim // 2 + pad_adjustment
	z_end_indices = index[:, 0] + neighborhood_dim // 2 + pad_adjustment
	
	neighborhood = []
	for i in range(index.shape[0]):
		xy_grid = full_grid_tensor_padded[index[:, 0][0], x_start_indices[i]:x_end_indices[i], y_start_indices[i]:y_end_indices[i], :].unsqueeze(0)
		if full_grid_tensor_padded.shape[0] > 1:
			xz_grid = full_grid_tensor_padded[z_start_indices[i]:z_end_indices[i], index[:, 1][0], y_start_indices[i]:y_end_indices[i], :].unsqueeze(0)
			yz_grid = full_grid_tensor_padded[z_start_indices[i]:z_end_indices[i], x_start_indices[i]:x_end_indices[i], index[:, 2][0], :].unsqueeze(0)
			neighborhood.append(torch.vstack([xy_grid, xz_grid, yz_grid]).unsqueeze(0))
		else:
			neighborhood.append(xy_grid.unsqueeze(0))
	try:
		neighborhood = torch.vstack(neighborhood)
	except:
		import pdb; pdb.set_trace()
	
	return neighborhood

def generate_indices(x, batch_size, shuffle=False):
	D, H, W = x.shape
	
	# Create indices for each dimension
	d_indices = torch.arange(D)
	h_indices = torch.arange(H)
	w_indices = torch.arange(W)

	# Create meshgrid
	d_mesh, h_mesh, w_mesh = torch.meshgrid(d_indices, h_indices, w_indices)

	# Flatten indices and concatenate
	indices = torch.stack((d_mesh.flatten(), h_mesh.flatten(), w_mesh.flatten()), dim=1)
	assert indices.shape == (D*H*W, 3) 
	
	if shuffle: 
		shuffled_idx = torch.randperm(D*H*W)
		indices = indices[shuffled_idx]

	true_indices = indices[x[indices.T[0], indices.T[1], indices.T[2]]]
	# make sure it's divisible by batch size 
	missing_indices = batch_size - (true_indices.shape[0] % batch_size)
	true_indices = torch.vstack([true_indices, true_indices[:missing_indices]])
	batched_indices=rearrange(true_indices, '(n b) c -> n b c',b=batch_size)
	return batched_indices

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

def custom_pad(tensor, neighborhood_dim):
	pad_adjustment = neighborhood_dim//2
	if tensor.shape[0] > 1:
		# In 3D case, pad D, H, W. Never pad batch dimension
		pad = (pad_adjustment, pad_adjustment, pad_adjustment, pad_adjustment, pad_adjustment, pad_adjustment)
		tensor = tensor.permute(3,0,1,2).unsqueeze(0)
		tensor_padded = F.pad(tensor, pad=pad, mode='circular')[0].permute(1,2,3,0)
	else: 
		# In 2D case, pad only H, W. Never pad batch dimension
		pad = (pad_adjustment, pad_adjustment, pad_adjustment, pad_adjustment)	
		tensor_padded = F.pad(tensor.permute(3,0,1,2), pad=pad, mode='circular').permute(1,2,3,0)

	return tensor_padded

def main(texture_file: str = 'zebra.png', 
		 object_file: str = 'cube.obj',
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
		 deterministic: bool = False, 
		 shuffle_indices: bool = True, 
		 display_freq: int = 4,
		 use_hist: bool = True,
		 experiment_name: str = None,
		 device: str = 'cpu',
		 ):
	
	
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
			(1, neighborhood_dim**2, neighborhood_dim**2, 3),
      	)
		mask = torch.ones_like(full_grid_tensor[:, :, :, 0]).bool()
	
	downsampled_full_grid = custom_interpolate(full_grid_tensor, scale_factor=resolutions[0])
	downsampled_mask = custom_interpolate(mask.float().unsqueeze(-1), scale_factor=resolutions[0]).bool().squeeze(-1)
	
	os.makedirs(f"gif/", exist_ok=True)
	os.makedirs(f"gif/{experiment_name}", exist_ok=True)
	os.makedirs(f'outputs/{experiment_name}/', exist_ok=True)
	os.makedirs(f'outputs/{experiment_name}/cross_sections', exist_ok=True)
	os.makedirs(f'outputs/{experiment_name}/voxel_grids', exist_ok=True)
	os.makedirs(f'outputs/{experiment_name}/final_outputs', exist_ok=True)
	os.makedirs(f'outputs/{experiment_name}/histograms', exist_ok=True)
 
	run_details = f"{texture_file.split('.')[0]}_{object_file.split('.')[0]}"
	for r in range(len(resolutions)):
		scale = resolutions[r]
		print(f"Commencing optimization at resolution {scale}")
		downsampled_texture = custom_interpolate(texture, scale_factor=scale)
		tex = downsampled_mask.shape[1:] if test_2d else downsampled_mask.shape
		if min(tex) <= neighborhood_dim:
			print(f"Skipping resolution {scale} (too downsampled)")
			print(f"Upsampling optimized tensor to resolution {resolutions[r+1]}")
			downsampled_full_grid = custom_interpolate(
				downsampled_full_grid, 
				scale_factor=int(resolutions[r+1]/resolutions[r]),
				mode='bicubic')
			downsampled_mask = custom_interpolate(
				downsampled_mask.float().unsqueeze(-1), 
				scale_factor=int(resolutions[r+1]/resolutions[r]),
				mode='bicubic').bool().squeeze(-1)
			continue
		tensor_show(downsampled_full_grid, show=show)
		if show:
			plt.imshow(downsampled_texture)
			plt.show()

		optimize = Optimize(downsampled_texture, r=r, use_hist=use_hist, device=device)
		search = Search(downsampled_texture, neighborhood_dim=neighborhood_dim, index=r, experiment_name=experiment_name)

		if deterministic:
			print("Generating deterministic indices")
			indices = generate_indices(downsampled_mask, batch_size, shuffle=shuffle_indices)

		batches = num_iters * indices.shape[0]
		for i in tqdm(range(batches)):
			downsampled_full_grid_padded = custom_pad(downsampled_full_grid, neighborhood_dim)
			if not deterministic:
				index = sample_voxel(downsampled_mask, batch_size=batch_size, neighborhood_dim=neighborhood_dim)
			else: 
				index = indices[i % indices.shape[0]]
			downsampled_full_grid_padded = custom_pad(downsampled_full_grid, neighborhood_dim)
			neighborhood = sample_neighborhood(downsampled_full_grid_padded, index, neighborhood_dim=neighborhood_dim)
			texel_match = search.find(neighborhood)
			
			new_value = optimize(exemplar=texel_match.to(device), solid=neighborhood.to(device))
			

			if (num_iters * indices.shape[0])//display_freq > 0 and i % ((num_iters * indices.shape[0])//display_freq) == 0:
				# For making the GIF
				interpolated = custom_interpolate(
					downsampled_full_grid, 
					scale_factor=int(resolutions[-1]/resolutions[r]),
					mode='bicubic')
				tensor_show(interpolated, show=False, filename=f"gif/{experiment_name}/res{r}_step{'{:05d}'.format(i)}.png")	
		
				difference= torch.norm(new_value.cpu() - downsampled_full_grid[index.T[0], index.T[1], index.T[2]])
				print(difference) 
				if difference < 1e-8:
					print("Skip to next resolution")
					continue
			
				torch.save(downsampled_full_grid, f"outputs/{experiment_name}/voxel_grids/{run_details}_{r}_{i}.pt")
				plt.imshow(downsampled_full_grid[downsampled_full_grid.shape[0]//2].cpu().numpy())
				plt.savefig(f'outputs/{experiment_name}/cross_sections/{run_details}_{r}_{i}.png')

				grid_show(texels=texel_match, voxels=neighborhood, show=show)
				tensor_show(downsampled_full_grid, show=show )	

			downsampled_full_grid[index.T[0], index.T[1], index.T[2]] = new_value.cpu()
			
							
		if r + 1 < len(resolutions):
			print(f"Upsampling optimized tensor to resolution {resolutions[r+1]}")
			downsampled_full_grid = custom_interpolate(
				downsampled_full_grid, 
				scale_factor=int(resolutions[r+1]/resolutions[r]),
				mode='bicubic')
			downsampled_mask = custom_interpolate(
				downsampled_mask.float().unsqueeze(-1), 
				scale_factor=int(resolutions[r+1]/resolutions[r]),
				mode='bicubic').bool().squeeze(-1)

	tensor_show(downsampled_full_grid, show=True)
	torch.save(downsampled_full_grid, f"outputs/{experiment_name}/voxel_grids/{run_details}_final.pt")
	if not test_2d:
		colors = pointify_tensor(full_grid_tensor, mask=mask)
	
	search.remove_cache()
 
	# convert colored voxel grid into a ply
	if not test_2d:
		ax = plt.figure().add_subplot(projection='3d')
		ax.voxels(full_grid.matrix,
				facecolors=full_grid_tensor.cpu().numpy(),
				linewidth=pitch)
		ax.set_aspect('equal')
		# Hide grid lines
		ax.grid(False)

		# Hide axes ticks
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_zticks([])
		plt.axis('off')

		ax.view_init(elev=30, azim=30)

		# Save multiple views
		angle_step_size = 45
		for theta in range(0, 360, angle_step_size):
			for phi in range(0, 180, angle_step_size):
				ax.view_init(elev=phi, azim=theta)
				plt.savefig(f"outputs/{experiment_name}/final_outputs/{theta}_{phi}.png")

		if show:
			plt.show()

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
		# server.add_point_cloud(
		# 	name="/full_grid",
		# 	points=full_grid_tensor[:, :, :, 0].nonzero().cpu().numpy() * pitch,
		# 	position=(0.0, 0.0, 0.0),
		# 	colors=(255, 0 , 0),
		# 	wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
		# 	point_size=0.01,
		# )

		server.add_point_cloud(
			name="/texture_voxels",
			points=full_grid.points,
			position=(0.0, 0.0, 0.0),
			colors=colors,
			wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
			point_size=pitch/2,
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