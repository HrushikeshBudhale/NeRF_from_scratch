import torch
import numpy as np
import torch.nn as nn

class NGP(nn.Module):
    def __init__(self, scene_scale: float=1.0, T: int=2**19, 
                 Nmin: int=16, Nmax: int=2048, Nlevels: int=16, 
                 L: int=4,  d: int=3, F: int=2, device=torch.device('cpu')):
        super().__init__()
        self.scene_scale = scene_scale                                                              # Downsizing factor to bring scene within [0, 1]^3
        self.T = T                                                                                  # Hash table size
        self.Nmin = Nmin                                                                            # Minimum grid resolution
        self.Nmax = Nmax                                                                            # Maximum grid resolution
        self.Nlevels = Nlevels                                                                      # Number of levels
        self.L = L                                                                                  # Number of frequencies for encoding
        self.d = d                                                                                  # grid dimension
        self.F = F                                                                                  # Number of feature channels
        self.device = device
        
        self.levels = np.geomspace(self.Nmin, self.Nmax, self.Nlevels, dtype=int)
        self.primes = torch.tensor([1, 2_654_435_761, 805_459_861], device=self.device)
        self._freqs = 2.0 ** torch.arange(self.L, dtype=torch.float32, device=self.device)          # (L)

        self.lookup_tables = torch.nn.ParameterDict(
            {str(i): nn.Parameter(
                   (torch.randn((T, F), device=self.device)*2-1)*1e-4
                ) for i in range(self.Nlevels)})                                                    # (NL, T, F)
        
        self.density_MLP = nn.Sequential(
            nn.Linear(self.F*self.Nlevels, 64), nn.ReLU(),
            nn.Linear(64, 16), 
        ).to(self.device)

        self.color_MLP = nn.Sequential(
            nn.Linear(16+self.d+(self.d*self.L*2), 64), nn.ReLU(),                                  # 16 from density_MLP + rest from encoding
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 3), nn.Sigmoid()
        ).to(self.device)

    def positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Npts, d)
        x_input = x.unsqueeze(-1) * (2 * torch.pi * self._freqs)                                    # (Npts, d, L)
        encoding = torch.cat([torch.sin(x_input), torch.cos(x_input)], dim=-1)                      # (Npts, d, L*2)
        encoding = torch.cat([x, encoding.reshape(*x.shape[:-1], -1)], dim=-1)                      # (Npts, d+d*L*2)
        return encoding

    def spatial_hashing(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Npts, 2**d, d)
        y = x * self.primes[:self.d]                                                                # (Npts, 2**d, d)
        if self.d == 2:
            z = torch.bitwise_xor(y[..., 0], y[..., 1])                                             # (Npts, 2**d)
        elif self.d == 3:
            z = torch.bitwise_xor(torch.bitwise_xor(y[..., 0], y[..., 1]), y[..., 2])
        hash = z % self.T                                                                           # (Npts, 2**d) 
        return hash

    def get_grid_vertices(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Npts, d)
        floor = torch.floor(x)                                                                      # (Npts, d)
        ceil  = torch.ceil(x)                                                                       # (Npts, d)
        vertices = torch.empty_like(x, dtype=torch.int64, device=self.device)
        vertices = vertices.unsqueeze(1).repeat(1, 2**self.d, 1)                                    # (Npts, 2**d, d)
        if self.d == 2: # square
            vertices[:, 0] = floor
            vertices[:, 1] = torch.cat([ ceil[:,[0]], floor[:,[1]]], dim=-1)
            vertices[:, 2] = torch.cat([floor[:,[0]],  ceil[:,[1]]], dim=-1)
            vertices[:, 3] = ceil
        if self.d == 3: # cube
            vertices[:, 0] = floor
            vertices[:, 1] = torch.cat([ ceil[:,[0]], floor[:,[1]], floor[:,[2]]], dim=-1)
            vertices[:, 2] = torch.cat([floor[:,[0]],  ceil[:,[1]], floor[:,[2]]], dim=-1)
            vertices[:, 3] = torch.cat([ ceil[:,[0]],  ceil[:,[1]], floor[:,[2]]], dim=-1)
            vertices[:, 4] = torch.cat([floor[:,[0]], floor[:,[1]],  ceil[:,[2]]], dim=-1)
            vertices[:, 5] = torch.cat([ ceil[:,[0]], floor[:,[1]],  ceil[:,[2]]], dim=-1)
            vertices[:, 6] = torch.cat([floor[:,[0]],  ceil[:,[1]],  ceil[:,[2]]], dim=-1)
            vertices[:, 7] = ceil

        return vertices                                                                             # (Npts, 2**d, d)

    def get_interpolated_features_from_table(self, x: torch.Tensor, 
                                             grid_vertices: torch.Tensor, 
                                             corner_features: torch.Tensor) -> torch.Tensor:
        # x: (Npts, d)
        # grid_vertices: (Npts, 2**d, d)
        # corner_features: (Npts, 2**d, F)
        corner_features = corner_features.swapaxes(1, 2)                                            # (Npts, F, 2**d)
        
        # points' position within each d dimensional unit cube centered at origin
        sample_position = (x - grid_vertices[:, 0]) - 0.5                                           # (Npts, d)
        
        # interpolation
        if self.d == 2:
            corner_features = corner_features.reshape(corner_features.shape[0], self.F,2,2)         # (Npts, F, 2, 2)
            sample_position = sample_position.unsqueeze(1).unsqueeze(1)                             # (Npts, 1, 1, d)
            features = torch.nn.functional.grid_sample(corner_features, sample_position,
                                                       mode='bilinear', align_corners=False)        # (Npts, F, 1, 1)
        elif self.d == 3:
            corner_features = corner_features.reshape(corner_features.shape[0], self.F,2,2,2)       # (Npts, F, 2, 2, 2)
            sample_position = sample_position.unsqueeze(1).unsqueeze(1).unsqueeze(1)                # (Npts, 1, 1, 1, d)
            features = torch.nn.functional.grid_sample(corner_features, sample_position, 
                                                       mode='bilinear', align_corners=False)        # (Npts, F, 1, 1, 1)

        return features.squeeze()                                                                   # (Npts, F)

    def forward(self, x: torch.Tensor, r_dir: torch.Tensor) -> torch.Tensor:
        # x    : (N, N_samples, d) point positions
        # r_dir: (N, N_samples, 3) ray directions
        N, N_samples = x.shape[0], r_dir.shape[1]
        x, r_dir = x.reshape(-1, self.d), r_dir.reshape(-1, 3)                                      # (N, d), (N, 3)
        color, sigma = torch.zeros_like(r_dir), torch.zeros_like(r_dir[:,[0]])                      # (N, 3), (N, 1)
        # bring scene within [0, 1]^d
        x = (x * self.scene_scale) + 0.5
        mask = torch.all(x > 0, dim=-1) & torch.all(x < 1, dim=-1)
        valid_x = x[mask]                                                                           # (Npts, d)
        
        features = torch.empty((valid_x.shape[0], self.F*self.Nlevels), device=self.device)         # (Npts, F*Nlevels)
        for i, level in enumerate(self.levels):
            level_scaled_pos = valid_x * level
            grid_vertices = self.get_grid_vertices(level_scaled_pos)                                # (Npts, 2**d, d)
            hashes = self.spatial_hashing(grid_vertices)                                            # (Npts, 2**d)
            corner_features = self.lookup_tables[str(i)][hashes]                                    # (Npts, 2**d, F)
            features[:, i*self.F:(i+1)*self.F] = \
                self.get_interpolated_features_from_table(level_scaled_pos, grid_vertices, 
                                                          corner_features)                          # (Npts, F)
        
        log_sigma = self.density_MLP(features)                                                      # (Npts, 16)
        encoded_dir = self.positional_encoding(r_dir[mask])                                         # (Npts, d+d*L*2)
        color[mask] = self.color_MLP(torch.cat([log_sigma, encoded_dir], dim=-1))                   # (N, 3)
        sigma[mask] = torch.exp(log_sigma[:, [0]])                                                  # (N, 1)

        color, sigma = color.reshape(N, N_samples, 3), sigma.reshape(N, N_samples, 1)               # (N, N_samples, 3), (N, N_samples, 1)
        return color, sigma                                                                         # (N, N_samples, 3), (N, N_samples, 1)