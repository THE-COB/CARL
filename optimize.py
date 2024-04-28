import torch
import einops 

class Optimize:
    def __init__(self, texture, r=0.8, num_bins=16,use_hist=True, device="cpu"):
        self.r = r
        self.texture = texture * 255.0
        self.use_hist = use_hist
        self.device = device
        if use_hist:
            self.num_bins = num_bins
            self.texture_hists = self.create_hist(texture.unsqueeze(0).unsqueeze(0).to(device))



    def __call__(self, exemplar: torch.Tensor, solid: torch.Tensor ) -> torch.Tensor:
        """
        solid:    torch.Tensor (n, 3, 8, 8, 3)
        exemplar: torch.Tensor (n, 3, 8, 8, 3)

        return:   torch.Tensor (n, 3)
        """
        exemplar = exemplar.clone() * 255.0
        exemplar = exemplar.to(self.device)
        solid = solid.clone() * 255.0
        solid = solid.to(self.device)
        # print(torch.max(exemplar), torch.max(solid))
        w = self.find_weight(exemplar, solid)
        if self.use_hist:
            w = self.update_weight(w, solid, exemplar)
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


    def update_weight(self, weight: torch.Tensor, solid: torch.Tensor, exemplar: torch.Tensor) -> torch.Tensor:
        """
        weight:   torch.Tensor (n, 3, 8, 8, 3)
        solid:    torch.Tensor (n, 3, 8, 8, 3)
        exemplar: torch.Tensor (n, 3, 8, 8, 3)

        return:   torch.Tensor (n, 3, 8, 8, 3)
        """
        # get bin number
        # solid_indices = solid//256 * self.num_bins
        exemplar_indices = (exemplar / 256 * self.num_bins).int()
        n, p, h, w, c = weight.shape
        solid_hists = self.create_hist(solid)
        try:
            hist_solid_values = solid_hists[[0, 1, 2]*n*h*w*p, exemplar_indices.flatten()].reshape(exemplar_indices.shape) 
        except:
            import pdb; pdb.set_trace()
        hist_exemplar_values = self.texture_hists[[0, 1,2]*n*h*w*p, exemplar_indices.flatten()].reshape(exemplar_indices.shape)
        diff = hist_solid_values - hist_exemplar_values
        hist_weights = weight / (1 + torch.sum(torch.where(diff > 0, diff, 0.0), dim=4, keepdim=True))
        return hist_weights
        

    

    
    def closed_form_irls(self, w: torch.Tensor, e: torch.Tensor):
        """
        w: torch.Tensor (n, 3, 8, 8, 3)
        e: torch.Tensor (n, 3, 8, 8, 3)

        returns s: torch.Tensor (n, 3)
        """

        s_new = 1/torch.sum(w, dim=(1,2,3)) * torch.sum(einops.rearrange(w * e, 'b p w h c -> b (p w h) c'),dim=1)

        return s_new
    
    def create_hist(self, patch: torch.Tensor) -> torch.Tensor:
        """
        patch: torch.Tensor (n, d, h, w, 3)
        channel: int [0, 1, 2] - represents the color chosen
        num_bins: int - default 16

        returns h: torch.Tensor (# bins)
        """

        # texels = texture[..., channel].view(-1)
        # h = torch.histogram(texels.float(), bins=num_bins, min=0, max=255)
        n, d, h, w, num_channels = patch.shape
        return torch.stack([(torch.histc(
                                patch[..., channel].view(-1).float(), 
                                bins=self.num_bins, 
                                min=0,
                                max=255,
                            )/d/h/w/n) for channel in range(num_channels)], dim=0)

