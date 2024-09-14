import torch


class Renderer:
    def __init__(self, conf: dict):
        self.near = conf["near"]
        self.far = conf["far"]
        self.step_size = (self.far - self.near) / conf["samples_per_ray"]
    
    def calc_weights(self, sigmas: torch.Tensor) -> torch.Tensor:
        # sigmas: (N_rays, N_samples, 1)
        N_rays = sigmas.shape[0]
        T_i = torch.exp(-self.step_size * torch.cumsum(sigmas, dim=1))                              # (N_rays, N_samples, 1)
        # transmittance of first sample is 1
        T_i = torch.cat([torch.ones((N_rays, 1, 1), device=sigmas.device), T_i[:, :-1]], dim=1)     # (N_rays, N_samples, 1)
        alpha = 1 - torch.exp(-sigmas * self.step_size)
        weights = alpha * T_i
        return weights # (N_rays, N_samples, 1)

    def volume_render(self, rgb: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        # rgb:    (N_rays, N_samples, 3)
        # sigmas: (N_rays, N_samples, 1)
        weights = self.calc_weights(sigmas)                                                             # (N_rays, N_samples, 1)
        rendered_colors = torch.sum(weights * rgb, dim=1) 
        return rendered_colors # (N_rays, 3)
    
    def volume_render_depth(self, sigmas: torch.Tensor) -> torch.Tensor:
        # sigmas: (N_rays, N_samples, 1)
        depths = torch.linspace(self.near, self.far, sigmas.shape[1], device=sigmas.device)
        weights = self.calc_weights(sigmas)                                                             # (N_rays, N_samples, 1)
        rendered_depths = torch.sum(weights * depths.unsqueeze(0).unsqueeze(-1), dim=1)
        return rendered_depths # (N_rays, 3)