import torch
import torchvision
import numpy as np
import trimesh
from torchvision.transforms.functional import pil_to_tensor

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

def sample_texture(texture, im):
	num_samples = im.shape[0] * im.shape[1]
	num_pixels = texture.shape[0] * texture.shape[1]
	p = torch.ones(num_pixels)/num_pixels
	index = torch.multinomial(input=p, num_samples=num_samples, replacement=True)
	init = texture.view((num_samples, -1))[index].reshape(im.shape)
	return init
