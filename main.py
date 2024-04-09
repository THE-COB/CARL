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
	# display mesh
	if show:
		plt.imshow( init )
		plt.show()
	
		server = viser.ViserServer()
		server.add_mesh_simple(
			name="/simple",
			vertices=vertices,
			faces=faces,
			wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
			position=(0.0, 0.0, 0.0),
		)
		
		while True:
			time.sleep(10.0)
	

if __name__ == '__main__':
	tyro.cli(main)