import torch
import torch.nn as nn



class Nerf(nn.Module):
    """
                        sigma
                        ^
                        |
    x -> block1 -> block2 -> block3 -> rgb
                   ^         ^
                   |         |
                   x         d

    """
    def __init__(self):
        super(Nerf, self).__init__()

        # Linear layers for xyz
        self.fc1_block1 = nn.Linear(2*3*10+3, 256)
        self.fc2_block1 = nn.Linear(256, 256)
        self.fc3_block1 = nn.Linear(256, 256)
        self.fc4_block1 = nn.Linear(256, 256)

        # Linear layers after concatenation
        self.fc1_block2 = nn.Linear(256+2*3*10+3, 256)
        self.fc2_block2 = nn.Linear(256, 256)
        self.fc3_block2 = nn.Linear(256, 256)
        self.fc4_block2 = nn.Linear(256, 256)
        
        # output layers
        self.linear_density = nn.Linear(256, 1)

        # Linear layers for rgb
        self.fc1_rgb = nn.Linear(256, 256)
        self.fc2_rgb = nn.Linear(256+2*3*4+3, 128)
        self.fc3_rgb = nn.Linear(128, 3)

        # Activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def positional_encoding(self, x: torch.Tensor, L: int):
        # x: (N_rays, N_samples, 3)
        freqs = 2.0 ** torch.arange(L, dtype=x.dtype).to(x.device)
        x_input = x.unsqueeze(-1) * freqs * 2 * torch.pi
        encoding = torch.cat([torch.sin(x_input), torch.cos(x_input)], dim=-1)
        encoding = torch.cat([x, encoding.reshape(*x.shape[:-1], -1)], dim=-1)
        return encoding

    def forward(self, x, r_d):
        # x: (N_rays, N_samples, 3)
        # r_d: (N_rays, N_samples, 3)
        
        x_encoded = self.positional_encoding(x, L=10) # (N_rays, N_samples, 2*3*10+3)
        r_d_encoded = self.positional_encoding(r_d, L=4) # (N_rays, N_samples, 2*3*4+3)

        x = self.relu(self.fc1_block1(x_encoded))
        x = self.relu(self.fc2_block1(x))
        x = self.relu(self.fc3_block1(x))
        x = self.relu(self.fc4_block1(x))
        
        # concatenate x again
        x = torch.cat([x, x_encoded], dim=-1)
        
        x = self.relu(self.fc1_block2(x))
        x = self.relu(self.fc2_block2(x))
        x = self.relu(self.fc3_block2(x))
        x = self.fc4_block2(x)
        
        # output density
        density = self.relu(self.linear_density(x))

        # process ray direction
        x = self.fc1_rgb(x)
        x = torch.cat([x, r_d_encoded], dim=-1)
        x = self.relu(self.fc2_rgb(x))
        rgb = self.sigmoid(self.fc3_rgb(x))

        return rgb, density
    

