from lightning_model import NuWave
from omegaconf import OmegaConf as OC
import os
import argparse
import datetime
from glob import glob
import torch
import librosa as rosa
from scipy.io.wavfile import write as swrite
import matplotlib.pyplot as plt
from utils.stft import STFTMag
import numpy as np
from filters import LowPass

def save_stft_mag(wav, fname):
    fig = plt.figure(figsize=(9, 3))
    plt.imshow(rosa.amplitude_to_db(stft(wav[0].detach().cpu()).numpy(),
               ref=np.max, top_db = 80.),
               aspect='auto',
               origin='lower',
               interpolation='none')
    plt.colorbar()
    plt.xlabel('Frames')
    plt.ylabel('Channels')
    plt.tight_layout()
    fig.savefig(fname, format='png')
    plt.close()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--checkpoint',
                        type=str,
                        required=True,
                        help="Checkpoint path")
    parser.add_argument('-f',
                        '--file_name',
                        type=str,
                        required=True,
                        help="File name")
    parser.add_argument('--sr',
                        type=int,
                        default=48000,
                        required=False,
                        help="Sampling rate for audio load")
    parser.add_argument('--steps',
                        type=int,
                        required=False,
                        help="Steps for sampling")
    parser.add_argument('--no_init_noise',
                        action='store_false',
                        default = True,
                        required=False,
                        help="no init noise ejection")
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        required=False,
                        help="Device, 'cuda' or 'cpu'")
    parser.add_argument('--max_save',
                        type=int,
                        default = 10,
                        required=False,
                        help="Maximum save samples")

    args = parser.parse_args()
    #torch.backends.cudnn.benchmark = False
    hparams = OC.load('hparameter.yaml')
    if args.steps is not None:
        hparams.ddpm.max_step = args.steps
        hparams.ddpm.noise_schedule = \
                "torch.tensor([1e-6,2e-6,1e-5,1e-4,1e-3,1e-2,1e-1,9e-1])"
    else:
        args.steps = hparams.ddpm.max_step
    args.max_save = min(args.steps, args.max_save)
    model = NuWave(hparams).to(args.device)
    stft = STFTMag()
    ckpt_path = os.path.join(hparams.log.checkpoint_dir, args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'] if not('EMA' in ckpt_path) else ckpt)
    wav, _ = rosa.load(args.file_name, sr=hparams.audio.sr, mono=True)
    wav = torch.Tensor(wav).unsqueeze(0).to(args.device)
    save_stft_mag(wav, f'{os.path.splitext(args.file_name)[0]}.png')
    
    lp = LowPass(ratio = [1/2]).to(args.device)
    wav = lp(wav, 0)
    swrite(f'{os.path.splitext(args.file_name)[0]}_{args.sr}.wav',
               hparams.audio.sr, wav[0].detach().cpu().numpy())

    save_stft_mag(wav, f'{os.path.splitext(args.file_name)[0]}_{args.sr}.png')
    upsampled = model.sample(wav, hparams.ddpm.max_step, args.no_init_noise,
                             True)
    #Plot, swrite etc. for later
    for i, uwav in enumerate(upsampled):
        t = hparams.ddpm.max_step - i
        if t>args.max_save:
            continue
        swrite(f'{os.path.splitext(args.file_name)[0]}_{t}.wav',
               hparams.audio.sr, uwav[0].detach().cpu().numpy())
        save_stft_mag(uwav, f'{os.path.splitext(args.file_name)[0]}_{t}.png')
        
        plt.close()
