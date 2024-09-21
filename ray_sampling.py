import torch

class RaySampler:
    def __init__(self, conf: dict):
        self.near = conf["near"]
        self.far = conf["far"]
        self.N_samples = conf["samples_per_ray"]
        self.Np_samples = conf["fine_samples_per_ray"]
        self.N_samples_fine = self.N_samples + self.Np_samples
        self.step_size = (self.far - self.near) / self.N_samples
        self.perturb_samples = conf["perturb_samples"]

    def sample_along_rays(self, r_o: torch.Tensor, r_d: torch.Tensor):
        """Stratified sampling along rays"""
        # r_o: (N_rays, 3)
        # r_d: (N_rays, 3)
        z_vals = torch.linspace(self.near, self.far, self.N_samples, device=r_o.device)                         # (N_samples)
        z_vals = z_vals.unsqueeze(0).repeat(r_o.shape[0], 1)                                                    # (N_rays, N_samples)
        mids = (z_vals[:, 1:] + z_vals[:, :-1]) / 2.0                                                           # (N_rays, N_samples-1)
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)                                                        # (N_rays, N_samples)
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)                                                       # (N_rays, N_samples)
        z_vals = lower + (upper - lower) * torch.rand(z_vals.shape, device=r_o.device) * self.perturb_samples   # (N_rays, N_samples)
        r_o, r_d = r_o.unsqueeze(1).repeat(1, self.N_samples, 1), r_d.unsqueeze(1).repeat(1, self.N_samples, 1) # (N_rays, N_samples, 3), (N_rays, N_samples, 3)
        points = r_o + r_d * z_vals.unsqueeze(-1)                                                               # (N_rays, N_samples, 3)
        return points, z_vals

    def hierarchical_sample_along_rays(self, r_o: torch.Tensor, r_d: torch.Tensor, z_vals: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Hierarchical sampling along rays"""
        # r_o: (N_rays, 3)
        # r_d: (N_rays, 3)
        # z_vals: (N_rays, N_samples) : sampled distances
        # weights: (N_rays, N_samples, 1) : weights of points
        
        # normalize weights
        weights = weights.detach()
        weights = weights[:, 1:-1, 0].squeeze(-1)                                                   # (N_rays, N_samples-2)
        weights = (weights + 1e-10) / torch.sum(weights + 1e-10, dim=-1, keepdim=True)              # (N_rays, N_samples-2)

        # calculate cdf
        cdf = torch.cumsum(weights, dim=-1)                                                         # (N_rays, N_samples-2)
        cdf = torch.cat([torch.zeros_like(cdf[:,:1]), cdf], dim=-1)                                 # (N_rays, N_samples-1)

        # sample points
        u = torch.linspace(0, 1, self.Np_samples+2, device=r_o.device)[1:-1]                        # (Np_samples)
        assert(u.min() > 0 and u.max() < 1)
        u = u.expand(cdf.shape[0], self.Np_samples)                                                 # (N_rays, Np_samples)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u)                                                           # (N_rays, Np_samples)
        assert(inds.min() >= 1 and inds.max() < self.N_samples-1)
        inds = torch.stack([inds-1, inds], dim=-1)                                                  # (N_rays, Np_samples, 2)

        new_shape =  list(inds.shape[:-1]) + [cdf.shape[-1]] # new_shape = (N_rays, Np_samples, N_samples-1)
        cdf = torch.gather(cdf.unsqueeze(-2).expand(new_shape), dim=-1, index=inds)                 # (N_rays, Np_samples, 2)

        t = (u - cdf[:,:, 0]) / (cdf[:,:, 1] - cdf[:,:, 0])                                         # (N_rays, Np_samples)
        assert(t.min() > 0 and t.max() <= 1)

        z_mids = (z_vals[:, 1:] + z_vals[:, :-1]) / 2.0                                             # (N_rays, N_samples-1)
        z_mids = torch.gather(z_mids.unsqueeze(-2).expand(new_shape), dim=-1, index=inds)           # (N_rays, Np_samples, 2)
        new_z_vals = z_mids[:,:, 0] + t * (z_mids[:,:, 1] - z_mids[:,:, 0])                         # (N_rays, Np_samples)

        z_combined = torch.sort(torch.cat([z_vals, new_z_vals], dim=-1), dim=-1)[0]                 # (N_rays, N_samples_fine)
        r_o = r_o.unsqueeze(1).repeat(1, self.N_samples_fine, 1)                                    # (N_rays, N_samples_fine, 3)
        r_d = r_d.unsqueeze(1).repeat(1, self.N_samples_fine, 1)                                    # (N_rays, N_samples_fine, 3)
        points = r_o + r_d * z_combined.unsqueeze(-1)                                               # (N_rays, N_samples_fine, 3)
        return points, z_combined                                                                   # (N_rays, N_samples_fine, 3), (N_rays, N_samples_fine)
