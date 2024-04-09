import torch

class Search:
	def __init__(self, exemplar, rank=20): 
		# precompute PCA of exemplar patches
		self.rank = rank
		self.set_pca_coordinates(exemplar)
		
	
	def set_pca_coordinates(self, exemplar):
		# following paper, 8x8 patches 
		samples = []
		for i in range(exemplar.shape[0] - 8):
			for j in range(exemplar.shape[0] - 8):
				samples.append(exemplar[i:i+8,j:j+8,:])
		matrix = torch.tensor(samples)
		U, S, V_T = torch.pca_lowrank(matrix, q=self.rank)
		proj = torch.matmul(samples, V_T.t[:, :self.rank])

