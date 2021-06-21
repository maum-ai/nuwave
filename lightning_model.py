#Some codes are adopted from
#https://github.com/ivanvovk/WaveGrad
#https://github.com/lmnt-com/diffwave
#https://github.com/lucidrains/denoising-diffusion-pytorch
#https://github.com/hojonathanho/diffusion
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.io.wavfile import write as swrite

from model import NuWave as model
import dataloader
from utils.tblogger import TensorBoardLoggerExpanded
import filters
from utils.stft import STFTMag


@torch.jit.script
def lognorm(pred, target):
    return (pred - target).abs().mean(dim=-1).clamp(min=1e-20).log().mean()


class NuWave(pl.LightningModule):
    def __init__(self, hparams, train=True):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = model(hparams)
        self.filter_ratio = [1. / hparams.audio.ratio]
        self.norm = nn.L1Loss()  #loss

        if not train:
            self.stft = STFTMag(2048, 512)

            def snr(pred, target):
                return (10 *torch.log10(torch.norm(target, dim=-1) \
                        /torch.norm(pred -target, dim =-1).clamp(min =1e-8))).mean()

            def lsd(pred, target):
                sp = torch.log(self.stft(pred).square().clamp(1e-8))
                st = torch.log(self.stft(target).square().clamp(1e-8))
                return (sp - st).square().mean(dim=1).sqrt().mean()

            self.snr = snr
            self.lsd = lsd

        self.set_noise_schedule(hparams, train)
    
    # DDPM backbone is adopted form https://github.com/ivanvovk/WaveGrad
    def set_noise_schedule(self, hparams, train=True):
        self.max_step = hparams.ddpm.max_step if train \
                else hparams.ddpm.infer_step
        noise_schedule = eval(hparams.ddpm.noise_schedule) if train \
                else eval(hparams.ddpm.infer_schedule)

        self.register_buffer('betas', noise_schedule, False)
        self.register_buffer('alphas', 1 - self.betas, False)
        self.register_buffer('alphas_cumprod', self.alphas.cumprod(dim=0),
                             False)
        self.register_buffer(
            'alphas_cumprod_prev',
            torch.cat([torch.FloatTensor([1.]), self.alphas_cumprod[:-1]]),
            False)
        alphas_cumprod_prev_with_last = torch.cat(
            [torch.FloatTensor([1.]), self.alphas_cumprod])
        self.register_buffer('sqrt_alphas_cumprod_prev',
                             alphas_cumprod_prev_with_last.sqrt(), False)
        self.register_buffer('sqrt_alphas_cumprod', self.alphas_cumprod.sqrt(),
                             False)
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             (1. / self.alphas_cumprod).sqrt(), False)
        self.register_buffer('sqrt_alphas_cumprod_m1',
                             (1. - self.alphas_cumprod).sqrt() *
                             self.sqrt_recip_alphas_cumprod, False)
        posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) \
                             / (1 - self.alphas_cumprod)
        posterior_variance = torch.stack(
                            [posterior_variance,
                             torch.FloatTensor([1e-20] * self.max_step)])
        posterior_log_variance_clipped = posterior_variance.max(
            dim=0).values.log()
        posterior_mean_coef1 = self.betas * self.alphas_cumprod_prev.sqrt() / (
            1 - self.alphas_cumprod)
        posterior_mean_coef2 = (1 - self.alphas_cumprod_prev
                                ) * self.alphas.sqrt() / (1 -
                                                          self.alphas_cumprod)
        self.register_buffer('posterior_log_variance_clipped',
                             posterior_log_variance_clipped, False)
        self.register_buffer('posterior_mean_coef1',
                             posterior_mean_coef1, False)
        self.register_buffer('posterior_mean_coef2',
                             posterior_mean_coef2, False)

    def sample_continuous_noise_level(self, step):
        rand = torch.rand_like(step, dtype=torch.float, device=step.device)
        continuous_sqrt_alpha_cumprod = \
                self.sqrt_alphas_cumprod_prev[step - 1] * rand \
                + self.sqrt_alphas_cumprod_prev[step] * (1. - rand)
        return continuous_sqrt_alpha_cumprod.unsqueeze(-1)

    def q_sample(self, y_0, step=None, noise_level=None, eps=None):
        batch_size = y_0.shape[0]
        if noise_level is not None:
            continuous_sqrt_alpha_cumprod = noise_level
        elif step is not None:
            continuous_sqrt_alpha_cumprod = self.sqrt_alphas_cumprod_prev[step]
        assert (step is not None or noise_level is not None)
        if isinstance(eps, type(None)):
            eps = torch.randn_like(y_0, device=y_0.device)
        outputs = continuous_sqrt_alpha_cumprod * y_0 + (
            1. - continuous_sqrt_alpha_cumprod**2).sqrt() * eps
        return outputs

    def q_posterior(self, y_0, y, step):
        posterior_mean = self.posterior_mean_coef1[step] * y_0  \
                         + self.posterior_mean_coef2[step] * y
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[step]
        return posterior_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def predict_start_from_noise(self, y, t, eps):
        return self.sqrt_recip_alphas_cumprod[t].unsqueeze(
            -1) * y - self.sqrt_alphas_cumprod_m1[t].unsqueeze(-1) * eps

    # t: interger not tensor
    @torch.no_grad()
    def p_mean_variance(self, y, y_down, t, clip_denoised: bool):
        batch_size = y.shape[0]
        noise_level = self.sqrt_alphas_cumprod_prev[t + 1].repeat(
            batch_size, 1)
        eps_recon = self.model(y, y_down, noise_level)
        y_recon = self.predict_start_from_noise(y, t, eps_recon)
        if clip_denoised:
            y_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance_clipped = self.q_posterior(
            y_recon, y, t)
        return model_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def compute_inverse_dynamincs(self, y, y_down, t, clip_denoised=False):
        model_mean, model_log_variance = self.p_mean_variance(
            y, y_down, t, clip_denoised)
        eps = torch.randn_like(y) if t > 0 else torch.zeros_like(y)
        return model_mean + eps * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def sample(self, y_down,
               start_step=None,
               init_noise=True,
               store_intermediate_states=False):
        batch_size = y_down.shape[0]
        start_step = self.max_step if start_step is None \
                else min(start_step, self.max_step)
        step = torch.tensor([start_step] * batch_size,
                            dtype=torch.long,
                            device=self.device)
        y_t = torch.randn_like(
                y_down, device=self.device) if init_noise \
                else self.q_sample(y_down, step=step)
        ys = [y_t]
        t = start_step - 1
        while t >= 0:
            y_t = self.compute_inverse_dynamincs(y_t, y_down, t)
            ys.append(y_t)
            t -= 1
        return ys if store_intermediate_states else ys[-1]

    def forward(self, x, x_clean, noise_level):
        x = self.model(x, x_clean, noise_level)
        return x

    def common_step(self, y, y_low, step):
        noise_level = self.sample_continuous_noise_level(step) \
                if self.training \
                else self.sqrt_alphas_cumprod_prev[step].unsqueeze(-1)
        eps = torch.randn_like(y, device=y.device)
        y_noisy = self.q_sample(y, noise_level=noise_level, eps=eps)
        eps_recon = self.model(y_noisy, y_low, noise_level)
        loss = lognorm(eps_recon, eps)
        return loss, y, y_low, y_noisy, eps, eps_recon

    def training_step(self, batch, batch_nb):
        wav, wav_l = batch
        step = torch.randint(
            0, self.max_step, (wav.shape[0], ), device=self.device) + 1
        loss, *_ = self.common_step(wav, wav_l, step)
        self.log('loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_nb):
        wav, wav_l = batch
        step = torch.randint(
            0, self.max_step, (wav.shape[0], ), device=self.device) + 1
        loss, y, y_low, y_noisy, eps, eps_recon = \
                self.common_step(wav, wav_l, step)

        self.log('val_loss', loss, sync_dist=True)
        if batch_nb == 0:
            i = torch.randint(0, wav.shape[0], (1, )).item()
            y_recon = self.predict_start_from_noise(y_noisy, step - 1,
                                                    eps_recon)
            eps_error = eps - eps_recon
            self.trainer.logger.log_spectrogram(y[i], y_low[i], y_noisy[i],
                                                y_recon[i], eps_error[i],
                                                step[i].item(),
                                                self.current_epoch)
            self.trainer.logger.log_audio(wav[i], y_low[i], y_noisy[i],
                                          y_recon[i], eps_error[i],
                                          self.current_epoch)

            
        return {
            'val_loss': loss,
        }

    def test_step(self, batch, batch_nb):
        wav, wav_l = batch
        wav_up = self.sample(wav_l, self.hparams.ddpm.infer_step)
        snr = self.snr(wav_up, wav)
        base_snr = self.snr(wav_l, wav)
        lsd = self.lsd(wav_up, wav)
        base_lsd = self.lsd(wav_l, wav)
        dict = {
            'snr': snr,
            'base_snr': base_snr,
            'lsd': lsd,
            'base_lsd': base_lsd,
            'snr^2': snr.pow(2),
            'base_snr^2': base_snr.pow(2),
            'lsd^2': lsd.pow(2),
            'base_lsd^2': base_lsd.pow(2)
        }
        if self.hparams.save:
            swrite(
                f'{self.hparams.log.test_result_dir}/test_{batch_nb}_up.wav',
                self.hparams.audio.sr, wav_up[0].detach().cpu().numpy())
            swrite(
                f'{self.hparams.log.test_result_dir}/test_{batch_nb}_orig.wav',
                self.hparams.audio.sr, wav[0].detach().cpu().numpy())
            swrite(
                f'{self.hparams.log.test_result_dir}/test_{batch_nb}_linear.wav',
                self.hparams.audio.sr, wav_l[0].detach().cpu().numpy())
            swrite(
                f'{self.hparams.log.test_result_dir}/test_{batch_nb}_down.wav',
                self.hparams.audio.sr // self.hparams.audio.ratio,
                wav_l[0, ::self.hparams.audio.ratio].detach().cpu().numpy())

        self.log_dict(dict)
        return dict

    def test_epoch_end(self, outputs):
        lsd = torch.stack([x['lsd'] for x in outputs]).mean()
        base_lsd = torch.stack([x['base_lsd'] for x in outputs]).mean()
        snr = torch.stack([x['snr'] for x in outputs]).mean()
        base_snr = torch.stack([x['base_snr'] for x in outputs]).mean()
        dict = {
            'snr': snr.item(),
            'base_snr': base_snr.item(),
            'lsd': lsd.item(),
            'base_lsd': base_lsd.item(),
        }
        print(dict)
        return

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(),
                               lr=self.hparams.train.lr,
                               eps=self.hparams.train.opt_eps,
                               betas=(self.hparams.train.beta1,
                                      self.hparams.train.beta2),
                               weight_decay=self.hparams.train.weight_decay)
        return opt

    def train_dataloader(self):
        return dataloader.create_vctk_dataloader(self.hparams, 0)

    def val_dataloader(self):
        return dataloader.create_vctk_dataloader(self.hparams, 1)

    def test_dataloader(self):
        return dataloader.create_vctk_dataloader(self.hparams, 2)
