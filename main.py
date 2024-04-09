import torch 
import torchvision
import matplotlib.pyplot as plt
import tyro
from pathlib import Path


def main(texture_file: str = 'tomatoes.png', 
		 object_file: str = "cube.obj", 
		 texture_dir: str = 'textures', 
		 object_dir: str = "objs", 
		 show: bool = False, 
		 device: str = 'cpu'):
	texture = torchvision.io.read_image(texture_dir + '/' + texture_file).permute(1, 2, 0)
	if show:
		plt.imshow( texture  )
		plt.show()

	num_samples = texture.shape[0] * texture.shape[1] 
	p = torch.ones(num_samples)/num_samples
	index = torch.multinomial(input=p, num_samples=num_samples, replacement=True)
	init = texture.view((num_samples, -1))[index].reshape(texture.shape)
	if show:
		plt.imshow( init )
		plt.show()

if __name__ == '__main__':
	tyro.cli(main)