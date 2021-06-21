#Some codes are adopted from 
#https://github.com/ivanvovk/WaveGrad
#https://github.com/lmnt-com/diffwave
#https://github.com/lucidrains/denoising-diffusion-pytorch
#https://github.com/hojonathanho/diffusion

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

Linear = nn.Linear
silu = F.silu

def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class DiffusionEmbedding(nn.Module):
    def __init__(self, hparam):
        super().__init__()
        self.n_channels = hparam.ddpm.pos_emb_channels
        self.scale = hparam.ddpm.pos_emb_scale
        self.out_channels = hparam.arch.pos_emb_dim
        half_dim = self.n_channels // 2
        exponents = torch.arange(half_dim, dtype=torch.float32)/ float(half_dim)
        exponents = 1e-4 ** exponents
        self.register_buffer('exponents', exponents)
        self.projection1 = Linear(self.n_channels, self.out_channels)
        self.projection2 = Linear(self.out_channels, self.out_channels)

    #noise_level: [B]
    def forward(self, noise_level):
        x = self.scale * noise_level * self.exponents.unsqueeze(0)
        x = torch.cat([x.sin(), x.cos()], dim=-1) #[B, self.n_channels]
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation, pos_emb_dim):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels,
                                   2*residual_channels,
                                   3,
                                   padding=dilation,
                                   dilation=dilation)
        self.diffusion_projection = Linear(pos_emb_dim, residual_channels)
        self.output_projection = Conv1d(residual_channels,
                                        2 * residual_channels, 1)
        self.low_projection = Conv1d(residual_channels,
                                     2*residual_channels,
                                     3,
                                     padding=dilation,
                                     dilation=dilation)



    def forward(self, x, x_low, noise_level):
        noise_level = self.diffusion_projection(noise_level).unsqueeze(-1)

        y = x + noise_level
        y = self.dilated_conv(y)
        y += self.low_projection(x_low)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class NuWave(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.input_projection = Conv1d(1, hparams.arch.residual_channels, 1)
        self.low_projection = Conv1d(1, hparams.arch.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(
            hparams)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(hparams.arch.residual_channels,
                          2**(i % hparams.arch.dilation_cycle_length),
                          hparams.arch.pos_emb_dim)
            for i in range(hparams.arch.residual_layers)
        ])
        self.len_res = len(self.residual_layers)
        self.skip_projection = Conv1d(hparams.arch.residual_channels,
                                      hparams.arch.residual_channels, 1)
        self.output_projection = Conv1d(hparams.arch.residual_channels, 1, 1)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, audio, audio_low, noise_level):
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = silu(x)
        x_low = self.low_projection(audio_low.unsqueeze(1))
        x_low = silu(x_low)
        noise_level = self.diffusion_embedding(noise_level)

        #This way is more faster!
        #skip = []
        skip =0.
        for layer in self.residual_layers:
            x, skip_connection = layer(x, x_low, noise_level)
            #skip.append(skip_connection)
            skip += skip_connection

        #x = torch.sum(torch.stack(skip), dim=0) / sqrt(self.len_res)
        x = skip / sqrt(self.len_res)
        x = self.skip_projection(x)
        x = silu(x)
        x = self.output_projection(x).squeeze(1)
        return x
