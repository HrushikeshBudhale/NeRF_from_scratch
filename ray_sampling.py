import torch

class RaySampler:
    def __init__(self, conf: dict):
        self.near = conf["near"]
        self.far = conf["far"]
        self.N_samples = conf["samples_per_ray"]
        self.step_size = (self.far - self.near) / self.N_samples
        self.perturb_samples = conf["perturb_samples"]

    def sample_along_rays(self, r_o: torch.Tensor, r_d: torch.Tensor):
        """Stratified sampling along rays"""
        # r_o: (N_rays, 3)
        # r_d: (N_rays, 3)
        z_vals = torch.linspace(self.near, self.far, self.N_samples, device=r_o.device)                         # (N_samples)
        z_vals = z_vals.unsqueeze(0).repeat(r_o.shape[0], 1)                                                    # (N_rays, N_samples)
        if self.perturb_samples:
            z_vals += torch.rand(r_o.shape[0], self.N_samples, device=r_o.device) * self.step_size
        r_o, r_d = r_o.unsqueeze(1).repeat(1, self.N_samples, 1), r_d.unsqueeze(1).repeat(1, self.N_samples, 1) # (N_rays, N_samples, 3), (N_rays, N_samples, 3)
        points = r_o + r_d * z_vals.unsqueeze(-1)                                                               # (N_rays, N_samples, 3)
        return points, z_vals