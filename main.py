import torch 
import torchvision
from torchvision.transforms.functional import pil_to_tensor
import matplotlib.pyplot as plt
import tyro
from pathlib import Path
import viser
import trimesh
import viser.transforms as tf
import numpy as np
import time
import pdb

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

def main(texture_file: str = 'tomatoes.png', 
		 object_file: str = "cube.obj", 
		 texture_dir: str = 'textures', 
		 object_dir: str = "objs", 
		 pitch: float = 0.01,
		 show: bool = True, 
		 device: str = 'cpu'):
	
	# Load and sample texture
	texture = torchvision.io.read_image(texture_dir + '/' + texture_file).permute(1, 2, 0)
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
	full_grid = trimesh.voxel.creation.voxelize(mesh, pitch=0.01).fill() #number of voxels in voxel grid
	end = time.time()
	print(f"Voxelized mesh with shape {full_grid.shape} in {end - start:.2f} seconds")
	num_samples = full_grid.points.shape[0]
	num_pixels = texture.shape[0] * texture.shape[1]
	p = torch.ones(num_pixels)/num_pixels
	index = torch.multinomial(input=p, num_samples=num_samples, replacement=True)
	full_colors = texture.view((num_pixels, -1))[index].reshape(full_grid.points.shape[0], 3) #gets colors of the sampled pixels and matches the shape of the mesh

	sample_voxels1, sample_colors1 = sample_voxels(np.array([-2.0,-1.0,-1.0]), 50, mesh, texture, pitch=0.01)
	sample_voxels2, sample_colors2 = sample_voxels(np.array([-2.0,-0.5,-1.0]), 50, mesh, texture, pitch=0.01)
	
	mesh_tensor = tensorize_mesh(mesh, pitch=0.01)
	print(mesh_tensor.shape)
	mesh_pointcloud = mesh_tensor.nonzero()[:, :3][::3]
	print(mesh_pointcloud.shape)
	# print(torch.sum(mesh_pointcloud))

	#for each voxel in the voxel grid:
		# for each axis (x, y, z)
			# for each neighbor
  
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
			points=full_grid.points,
			position=(0.0, 0.0, 0.0),
			colors=full_colors,
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