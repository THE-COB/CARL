import torch
import torchvision
import numpy as np
import trimesh
from torchvision.transforms.functional import pil_to_tensor
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def sample_voxels(sample_point, radius, mesh, texture, pitch=0.01):
	# Sample voxel mesh
	sample_grid = trimesh.voxel.creation.local_voxelize(mesh=mesh, point=sample_point, pitch=pitch, radius=radius).apply_scale(pitch)
	num_samples = sample_grid.points.shape[0]
	num_pixels = texture.shape[0] * texture.shape[1]
	p = torch.ones(num_pixels)/num_pixels
	index = torch.multinomial(input=p, num_samples=num_samples, replacement=True)
	sample_colors = texture.view((num_pixels, -1))[index].reshape(sample_grid.points.shape[0], 3).numpy()
	return sample_grid, sample_colors

def tensorize_mesh(mesh, pitch=0.01):
	num_points = int((mesh.bounds[1][0] - mesh.bounds[0][0]) / pitch)
	heights = np.linspace(mesh.bounds[0][0], mesh.bounds[1][0], num_points)
	max_width = mesh.bounds[1][1] - mesh.bounds[0][1]
	max_length = mesh.bounds[1][2] - mesh.bounds[0][2]

	sections = mesh.section_multiplane(plane_origin=mesh.centroid, plane_normal=[1,0,0], heights=heights)
	planes = []
	max_width = 0
	max_length = 0
	for s in sections:
		if s is not None:
			new_plane = pil_to_tensor(s.rasterize(pitch=pitch))[0]
			max_width = max(max_width, new_plane.shape[1])
			max_length = max(max_length, new_plane.shape[0])
			planes.append(new_plane)
	for i in range(len(planes)):
		width_pad_left = (max_width - planes[i].shape[1]) // 2
		length_pad_top = (max_length - planes[i].shape[0]) // 2
		width_pad_right = max_width - planes[i].shape[1] - width_pad_left
		length_pad_bottom = max_length - planes[i].shape[0] - length_pad_top
		planes[i] = torch.nn.functional.pad(planes[i], (width_pad_left, width_pad_right, length_pad_top, length_pad_bottom))
	planes = torch.stack(planes).unsqueeze(3)
	# make planes rgb
	planes = planes.repeat(1, 1, 1, 3)
	return planes

def sample_texture(texture, im_shape, device='cpu'):
	im_shape = (4 - len(im_shape)) * (1,) + im_shape
	num_samples = im_shape[1] * im_shape[2] 
	num_pixels = texture.shape[0] * texture.shape[1]
	p = torch.ones(num_pixels)/num_pixels # Initialize a uniform distribution over samples
	p = p.to(device)
	index = torch.multinomial(input=p, num_samples=num_samples, replacement=True).to(device) #samples from uniform distribution
	init = texture.view((-1, 3))[index].reshape(im_shape) # gets colors of sampled pixels from texture image + shapes into texture shape
	return init

def grid_show(texels, voxels, show):
	if show:
		fig = plt.figure(figsize=(4., 4.))
		grid = ImageGrid(fig, 111,  # similar to subplot(111)
										nrows_ncols=(2,4),  # creates 2x2 grid of axes
										axes_pad=0.1,  # pad between axes in inch.
										)

		for ax, im in zip(grid, list(voxels[:4]) + list(texels[:4])):
				# Iterating over the grid returns the Axes.
				ax.imshow(im[0])
		plt.show()

def tensor_show(tensor, show,filename=None):
	slice = torch.randint(tensor.shape[0], (1,))[0]
	if show:
		plt.imshow(tensor[slice].cpu().numpy())
		plt.show()
	if filename:
		plt.imsave(f"gif/{filename}", tensor[slice].cpu().numpy())

def pointify_tensor(full_grid_tensor: torch.Tensor, mask: torch.Tensor):
	"""
	full_grid_tensor:  torch.Tensor (d, h, w, 3)
	mask:  torch.Tensor (d, h, w)

	return: points: colors
	"""
	colors = full_grid_tensor[mask].view(-1, 3)
	return colors
