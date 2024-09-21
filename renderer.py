import torch


class Renderer:
    def __init__(self, conf: dict):
        self.near = conf["near"]
        self.far = conf["far"]
        self.step_size = (self.far - self.near) / conf["samples_per_ray"]
    
    def calc_weights(self, sigmas: torch.Tensor, z_vals: torch.Tensor) -> torch.Tensor:
        # sigmas: (N_rays, N_samples, 1)
        # z_vals: (N_rays, N_samples)
        N_rays = sigmas.shape[0]
        
        dists = (z_vals[:,1:] - z_vals[:,:-1]).unsqueeze(-1)                                              # (N_rays, N_samples-1, 1)
        dists = torch.cat([dists, 1e10 * torch.ones_like(dists[:, :1, :])], dim=1)                        # (N_rays, N_samples, 1)
        alpha = 1 - torch.exp(-sigmas * dists)                                                            # (N_rays, N_samples, 1)
        
        # transmittance: A cumulative product expresses the 
        # probability of ray transmitting through each sample
        T_i = torch.cumprod((1-alpha),1)                                                                  # (N_rays, N_samples, 1)
        T_i = torch.cat([torch.ones((N_rays, 1, 1), device=sigmas.device), T_i[:, :-1]], dim=1)           # (N_rays, N_samples, 1)
        weights = alpha * T_i
        return weights # (N_rays, N_samples, 1)

    def volume_render(self, rgb: torch.Tensor, sigmas: torch.Tensor, z_vals: torch.Tensor) -> torch.Tensor:
        # rgb:    (N_rays, N_samples, 3)
        # sigmas: (N_rays, N_samples, 1)
        # z_vals: (N_rays, N_samples)
        weights = self.calc_weights(sigmas, z_vals)                                                       # (N_rays, N_samples, 1)
        rendered_colors = torch.sum(weights * rgb, dim=1) 
        return rendered_colors # (N_rays, 3)

    def volume_render_depth(self, sigmas: torch.Tensor, z_vals: torch.Tensor) -> torch.Tensor:
        # sigmas: (N_rays, N_samples, 1)
        # z_vals: (N_rays, N_samples)
        weights = self.calc_weights(sigmas, z_vals)                                                       # (N_rays, N_samples, 1)
        z_vals = (z_vals - self.near) / (self.far - self.near)
        z_vals = z_vals.unsqueeze(-1).repeat(1, 1, 3)                                                     # (N_rays, N_samples, 3)
        rendered_depths = torch.sum(weights * z_vals, dim=1)                                              # (N_rays, 3)
        return rendered_depths # (N_rays, 3)