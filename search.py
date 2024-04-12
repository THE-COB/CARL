import torch
import torchvision
from annlite import AnnLite
from docarray import DocumentArray
#import pynndescent
import einops 
import matplotlib.pyplot as plt

class Search:
	def __init__(self, exemplar, rank=5): 
		# precompute PCA of exemplar patches
		self.rank = rank
		self.set_pca_coordinates(exemplar)
		
	
	def set_pca_coordinates(self, exemplar):
		# following paper, 8x8 patches 
		samples = []
		for i in range(exemplar.shape[0] - 8):
			for j in range(exemplar.shape[0] - 8):
				samples.append(exemplar[i:i+8,j:j+8,:].flatten().float()) 
		matrix = torch.vstack(samples) # num_texel_patches x 192
		U, S, V_T = torch.pca_lowrank(matrix, q=self.rank)
		self.V_T = V_T.float()
		self.texel_embeddings = torch.matmul(matrix, self.V_T) # num_texel_patches x 20
		self.pca_to_samples = {}
		for idx, sample in enumerate(samples):
			self.pca_to_samples[self.texel_embeddings[idx]] = sample

	def ann(self, voxel_embeddings):
		ann = AnnLite(self.rank, metric='cosine', data_path="/tmp/annlite_data")
		docs = DocumentArray.empty(self.texel_embeddings.shape[0])
		docs.embeddings = self.texel_embeddings
		ann.index(docs)

		query = DocumentArray.empty(voxel_embeddings.shape[0])
		query.embeddings = voxel_embeddings
		ann.search(query)
		examples = torch.vstack(list(self.pca_to_samples.keys()))

		matches = []
		for q in query:
			embedding = ann.get_doc_by_id(q.matches[0].id).embedding
			
			match = self.pca_to_samples[list(self.pca_to_samples.keys())[(embedding == examples).all(axis=1).nonzero()]]
			matches.append(match)
		return torch.vstack(matches)

		
	def pnn(self, voxel_patches):
		#Experimenting with pynndescent?
		indices = pynndescent.NNDescent(voxel_patches)
		indices.prepare()
		neighbors = indices.neighbor_graph[0]
		

	def find(self, voxel_patches, nn_method='ann'):
		n, p, h, w, c = voxel_patches.shape
		voxel_patches = einops.rearrange(voxel_patches, 'n p h w c -> (n p) (h w c)').float() # N*3 x 192
		voxel_embeddings = torch.matmul(voxel_patches, self.V_T)
		if nn_method == 'ann':
			matches = self.ann(voxel_embeddings)
		elif nn_method == 'pnn':
			matches = self.pnn(voxel_patches)
		else: 
			raise Exception("better luck next time")
		matches = einops.rearrange(matches, '(n p) (h w c) -> n p h w c', n=n, p=p, h=h, w=w, c=c)
		return matches



if __name__ == "__main__":
	texture_file = 'tomatoes.png'
	texture_dir = 'textures'
	texture = torchvision.io.read_image(texture_dir + '/' + texture_file).permute(1, 2, 0)[:18]
	search = Search(exemplar=texture)

	dummy_voxels = texture[:8,:8, :].unsqueeze(0).unsqueeze(0)
	plt.imshow(texture[:8,:8, :])
	plt.show()
	matches = search.find(dummy_voxels)
	plt.imshow(matches[0][0].int())
	plt.show()
	

	
	
