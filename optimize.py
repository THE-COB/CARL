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
    
    def closed_form_irls(self, w: torch.Tensor, e: torch.Tensor):
        """
        w: torch.Tensor (n, 3)
        e: torch.Tensor (n, 3, 8, 8, 3)

        returns s: torch.Tensor (n, 3)
        """
        s_new = torch.sum(einops.rearrange(w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * e, 'b p w h c -> b (p w h) c'),dim=1)\
            /torch.sum(w)

        return s_new

