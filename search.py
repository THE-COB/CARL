import torch
import torchvision
from annlite import AnnLite
from docarray import DocumentArray
#import pynndescent
import einops 
import matplotlib.pyplot as plt
import shutil 
import os

class Search:
	def __init__(self, exemplar, rank=5, neighborhood_dim=8): 
		# precompute PCA of exemplar patches
		self.rank = rank
		self.neighborhood_dim = neighborhood_dim
		self.set_pca_coordinates(exemplar)
		
	def set_pca_coordinates(self, exemplar):
		# following paper, 8x8 patches 
		samples = []
		dim = self.neighborhood_dim
		for i in range(exemplar.shape[0] - dim):
			for j in range(exemplar.shape[0] - dim):
				samples.append(exemplar[i:i+dim,j:j+dim,:].flatten().float()) 
		matrix = torch.vstack(samples) # num_texel_patches x 192
		_, _, V_T = torch.pca_lowrank(matrix, q=self.rank)
		self.V_T = V_T.float()
		self.texel_embeddings = torch.matmul(matrix, self.V_T) # num_texel_patches x 20
		self.pca_to_samples = {}
		for idx, sample in enumerate(samples):
			self.pca_to_samples[self.texel_embeddings[idx]] = sample
		
		# Necessary to empty tmp because otherwise previous embeddings will be included 
		if os.path.isdir('tmp/'):
			for dir_ in os.listdir('tmp/'):
				shutil.rmtree(f'tmp/{dir_}')
		self.ann = AnnLite(self.rank, metric='cosine', data_path="tmp/")
		docs = DocumentArray.empty(self.texel_embeddings.shape[0])
		docs.embeddings = self.texel_embeddings
		self.ann.index(docs)

	def approximate_nearest_neighbors(self, voxel_embeddings):
		voxel_embeddings = voxel_embeddings
		query = DocumentArray.empty(voxel_embeddings.shape[0])
		query.embeddings = voxel_embeddings
		self.ann.search(query)
		examples = torch.vstack(list(self.pca_to_samples.keys()))

		matches = []
		for q in query:
			embedding = q.matches[0].embedding
			match = self.pca_to_samples[list(self.pca_to_samples.keys())[(embedding == examples).all(axis=1).nonzero()]]
			matches.append(match)
		return torch.vstack(matches)

		
	def find(self, voxel_patches):
		n, p, h, w, c = voxel_patches.shape
		voxel_patches = einops.rearrange(voxel_patches, 'n p h w c -> (n p) (h w c)').float() # N*3 x 192
		voxel_embeddings = torch.matmul(voxel_patches, self.V_T)
		matches = self.approximate_nearest_neighbors(voxel_embeddings)
		
		matches = einops.rearrange(matches, '(n p) (h w c) -> n p h w c', n=n, p=p, h=h, w=w, c=c)
		return matches



if __name__ == "__main__":
	texture_file = 'tomatoes.png'
	texture_dir = 'textures'
	texture = torchvision.io.read_image(texture_dir + '/' + texture_file).permute(1, 2, 0)
	search = Search(exemplar=texture)

	dummy_voxels = texture[:8,:8, :].unsqueeze(0).unsqueeze(0)

	dummy_voxels = torch.randn([4, 3, 8, 8, 3])
	plt.imshow(dummy_voxels[0,0,:8,:8, :])
	plt.show()
	matches = search.find(dummy_voxels)
	plt.imshow(matches[0][0].int())
	plt.show()
	

	
	
