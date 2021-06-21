from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from os import path, makedirs
from omegaconf import OmegaConf as OC
from datetime import datetime, timedelta
from utils.stft import STFTMag
import librosa as rosa

class TensorBoardLoggerExpanded(TensorBoardLogger):
    def __init__(self, hparam):
        super().__init__(hparam.log.tensorboard_dir, name=hparam.name,
                default_hp_metric= False)
        self.hparam = hparam
        self.log_hyperparams(hparam)
        
        self.stftmag = STFTMag()

    def fig2np(self, fig):
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        return data

    def plot_spectrogram_to_numpy(self, y, y_low, y_noisy, y_recon,
                                  eps_error, step):

        name_list = ['y', 'y_low', 'y_noisy', 'y_recon','errer_recon']
        fig = plt.figure(figsize=(9, 15))
        fig.suptitle(f'Diffstep_{step}')
        for i, yy in enumerate([y, y_low, y_noisy, y_recon, eps_error]):
            ax=plt.subplot(5, 1, i + 1)
            ax.set_title(name_list[i])
            plt.imshow(rosa.amplitude_to_db(self.stftmag(yy).numpy(),
                       ref=np.max,top_db=80.),
                       #vmin = -20,
                       vmax = 0.,
                       aspect='auto',
                       origin='lower',
                       interpolation='none')
            plt.colorbar()
            plt.xlabel('Frames')
            plt.ylabel('Channels')
            plt.tight_layout()

        fig.canvas.draw()
        data = self.fig2np(fig)

        plt.close()
        return data

    @rank_zero_only
    def log_spectrogram(self, y, y_low, y_noisy, y_recon, eps_error,
                        diff_step, epoch):
        y, y_low, y_noisy, y_recon, eps_error = y.detach().cpu(
        ), y_low.detach().cpu(), y_noisy.detach().cpu(
        ), y_recon.detach().cpu(), eps_error.detach().cpu()
        spec_img = self.plot_spectrogram_to_numpy(
                y, y_low, y_noisy, y_recon,
                eps_error, diff_step)
        self.experiment.add_image(path.join(self.save_dir, 'result'),
                                  spec_img,
                                  epoch,
                                  dataformats='HWC')
        self.experiment.flush()
        return

    @rank_zero_only
    def log_audio(self, y, y_low, y_noisy, y_recon, eps_error,
                        epoch):
        y, y_low, y_noisy, y_recon, eps_error = y.detach().cpu(
        ), y_low.detach().cpu(), y_noisy.detach().cpu(
        ), y_recon.detach().cpu(), eps_error.detach().cpu()


        name_list = ['y', 'y_low', 'y_noisy', 'y_recon','errer_recon']

        for n, yy in zip(name_list, [y, y_low, y_noisy, y_recon, eps_error]):
            self.experiment.add_audio(n,
                                      yy, epoch, self.hparam.audio.sr)
        
        self.experiment.flush()
        return

    
