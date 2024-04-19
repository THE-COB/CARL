import torch
import einops 

class Optimize:
    def __init__(self, texture, r=0.8, num_bins=16):
        self.r = r
        self.texture = texture
        self.texture_hists = torch.stack([self.create_texture_hist(texture, 0), self.create_texture_hist(texture, 1), self.create_texture_hist(texture, 2)], dim=0)
        self.num_bins = num_bins
        self.solid_hists = torch.stack

    def __call__(self, exemplar: torch.Tensor, solid: torch.Tensor, tol=0.001, max_iter=1000) -> torch.Tensor:
        """
        solid:    torch.Tensor (n, 3, 8, 8, 3)
        exemplar: torch.Tensor (n, 3, 8, 8, 3)

        return:   torch.Tensor (n, 3)
        """
        
        w = self.find_weight(exemplar, solid)
        s = self.closed_form_irls(w, exemplar)
        
        return s
        

    def find_weight(self, exemplar: torch.Tensor, solid: torch.Tensor) -> torch.Tensor:
        """
        solid:    torch.Tensor (n, 3, 8, 8, 3)
        exemplar: torch.Tensor (n, 3, 8, 8, 3)

        return:   torch.Tensor (n, 3)
        """
        
        #subtract solid - exemplar
        diff = solid - exemplar

        #find norm of this value
        n = torch.norm(diff.flatten(2), dim=2) # [n, num_axes]

        #raise to power of r - 2
        return torch.pow(n, (self.r - 2)) # [n, num_axes]


    def update_weight(self, weight: torch.Tensor, solid: torch.Tensor, exemplar: torch.Tensor) -> torch.Tensor:
        """
        weight:   torch.Tensor (n, 3)
        solid:    torch.Tensor (n, 3, 8, 8, 3)
        exemplar: torch.Tensor (n, 3, 8, 8, 3)

        return:   torch.Tensor (n, 3)
        """
    # get bin number
        bin_width = 1/self.num_bins
        bin_indices = weight//bin_width #(n, 3)
        n, c = bin_indices.shape
        hist_values = self.texture_hists[[0, 1, 2]*n, bin_indices.flatten()].shape(bin_indices.shape) #(n,3)
        

    
        

    
    def closed_form_irls(self, w: torch.Tensor, e: torch.Tensor):
        """
        w: torch.Tensor (n, 3)
        e: torch.Tensor (n, 3, 8, 8, 3)

        returns s: torch.Tensor (n, 3)
        """
        s_new = torch.sum(einops.rearrange(w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * e, 'b p w h c -> b (p w h) c'),dim=1)\
            /torch.sum(w)

        return s_new
    
    def create_texture_hist(self, texture: torch.Tensor, channel: int, num_bins: int) -> torch.Tensor:
        """
        texture: torch.Tensor (h, w, 3)
        channel: int [0, 1, 2] - represents the color chosen
        num_bins: int - default 16

        returns h: torch.Tensor (# bins)
        """
        texels = texture[..., channel].view(-1)
        h = torch.histogram(texels.float(), bins=num_bins, min=0, max=255)

        return h
    
    def create_solid_hist(self, solid: torch.Tensor, channel: int, num_bins: int) -> torch.Tensor:
        


