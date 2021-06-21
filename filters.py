import torch
import torch.nn as nn
import torch.nn.functional as F

class LowPass(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256,
                 ratio=(1 / 6, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 4 / 5, 5 / 6,
                        1 / 1)):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)
        f = torch.ones((len(ratio), nfft//2 + 1), dtype=torch.float)
        for i, r in enumerate(ratio):
            f[i, int((nfft//2+1) * r):] = 0.
        self.register_buffer('filters', f, False)

    #x: [B,T], r: [B], int
    @torch.no_grad()
    def forward(self, x, r):
        if x.dim()==1:
            x = x.unsqueeze(0)
        T = x.shape[1]
        x = F.pad(x, (0, self.nfft), 'constant', 0)
        stft = torch.stft(x,
                          self.nfft,
                          self.hop,
                          window=self.window,
                          )#return_complex=False)  #[B, F, TT,2]
        stft *= self.filters[r].view(*stft.shape[0:2],1,1 )
        x = torch.istft(stft,
                        self.nfft,
                        self.hop,
                        window=self.window,
                        )#return_complex=False)
        x = x[:, :T].detach()
        return x


class HighPass(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256,
                 ratio=(1 / 6, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 4 / 5, 5 / 6,
                        1 / 1)):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)
        f = torch.ones((len(ratio), nfft//2 + 1), dtype=torch.float)
        for i, r in enumerate(ratio):
            f[i, :int((nfft//2+1) * r)] = 0.
        self.register_buffer('filters', f, False)

    #x: [B,T], r: [B], int
    @torch.no_grad()
    def forward(self, x, r):
        if x.dim()==1:
            x = x.unsqueeze(0)
        T = x.shape[1]
        x = F.pad(x, (0, self.nfft), 'constant', 0)
        stft = torch.stft(x,
                          self.nfft,
                          self.hop,
                          window=self.window,
                          )#return_complex=False)  #[B, F, TT,2]
        stft *= self.filters[r].view(*stft.shape[0:2],1,1 )
        x = torch.istft(stft,
                        self.nfft,
                        self.hop,
                        window=self.window,
                        )#return_complex=False)
        x = x[:, :T].detach()
        return x
