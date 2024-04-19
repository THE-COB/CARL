import torch
import einops 

class Optimize:
    def __init__(self, r=0.8):
        self.r = r

    def __call__(self, exemplar: torch.Tensor, solid: torch.Tensor, tol=0.001, max_iter=1000) -> torch.Tensor:
        """
        solid:    torch.Tensor (n, 3, 8, 8, 3)
        exemplar: torch.Tensor (n, 3, 8, 8, 3)

        return:   torch.Tensor (n, 3)
        """
        exemplar = exemplar.clone() * 255.0
        solid = solid.clone() * 255.0
        # print(torch.max(exemplar), torch.max(solid))
        w = self.find_weight(exemplar, solid)
        s = self.closed_form_irls(w, exemplar)
        
        # print(f"s min and max, {torch.min(s/255)}, {torch.max(s/255)}")
        return s / 255.0
        

    def find_weight(self, exemplar: torch.Tensor, solid: torch.Tensor) -> torch.Tensor:
        """
        solid:    torch.Tensor (n, 3, 8, 8, 3)
        exemplar: torch.Tensor (n, 3, 8, 8, 3)

        return:   torch.Tensor (n, 3, 8, 8, 3)
        """
        
        #subtract solid - exemplar
        diff = solid - exemplar
        
        #find norm of this value
        b, p, h, w, c = solid.shape 
        
        n = torch.norm(einops.rearrange(diff, 'b p h w c -> b p (h w) c'), dim=2) # [n, num_axes, channel]
        n = einops.rearrange(torch.repeat_interleave(n.unsqueeze(2), h*w, dim=2), 'b p (h w) c -> b p h w c', h=h, w=w)

        return torch.pow(n, (self.r - 2)) # [n, num_axes, 1, 1, channel]
    
    def closed_form_irls(self, w: torch.Tensor, e: torch.Tensor):
        """
        w: torch.Tensor (n, 3, 8, 8, 3)
        e: torch.Tensor (n, 3, 8, 8, 3)

        returns s: torch.Tensor (n, 3)
        """

        s_new = 1/torch.sum(w, dim=(1,2,3)) * torch.sum(einops.rearrange(w * e, 'b p w h c -> b (p w h) c'),dim=1)

        return s_new

