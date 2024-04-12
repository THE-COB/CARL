import torch
from torch.linalg.vector import norm

class Optimize:
    def __init__(self, r=0.8):
        self.r = r

    def optimize(self, exemplar: torch.Tensor, solid: torch.Tensor, tol=0.001, max_iter=1000) -> torch.Tensor:
        """
        solid:    torch.Tensor (n, 3, 8, 8, 3)
        exemplar: torch.Tensor (n, 3, 8, 8, 3)

        return:   torch.Tensor (n, 3, 8, 8, 3)
        """
        
        for i in range (max_iter):
            w = self.find_weight(exemplar, solid)
            s = self.closed_form_irls(w, exemplar)
            if torch.norm(s - solid) < tol:
                solid = s
                break
            solid = s
        
        return s
        

    def find_weight(self, exemplar: torch.Tensor, solid: torch.Tensor) -> torch.Tensor:
        """
        solid:    torch.Tensor (n, 3, 8, 8, 3)
        exemplar: torch.Tensor (n, 3, 8, 8, 3)

        return:   torch.Tensor (n, 3, 3)
        """
        #subtract solid - exemplar
        diff = solid - exemplar

        #find norm of this value
        n = norm(diff, dim=(2,3)) # [n, 3, 3]

        #raise to power of r - 2
        return torch.pow(n, (self.r - 2)) # [n, 3, 3]
    
    def closed_form_irls(self, w: torch.Tensor, e: torch.Tensor):
        """
        w: torch.Tensor (n, 3, 3)
        e: torch.Tensor (n, 3, 8, 8, 3)

        returns s: torch.Tensor (n, 3, 8, 8, 3)
        """

        w_new = w.unsqueeze(-2).unsqueeze(-3)
        w_broadcasted = w_new.expand(-1, -1, 8, 8, -1)

        s_new = torch.sum(torch.sum(w_broadcasted * e, dim=(2,3), keepdim=True), dim=1, keep_dim=True) / torch.sum(torch.sum(w_broadcasted, dim=(2,3), keepdim=True), dim=1, keep_dim=True)

        return s_new

