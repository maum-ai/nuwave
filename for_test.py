from lightning_model import NuWave
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf as OC
import os
import argparse
import datetime
from glob import glob
import torch
from tqdm import tqdm
from scipy.io.wavfile import write as swrite
def test(args):
    hparams = OC.load('hparameter.yaml')
    hparams.save = args.save or False
    model = NuWave(hparams, False).cuda()
    if args.ema:
        ckpt_path = glob(os.path.join(hparams.log.checkpoint_dir,
                         f'*_epoch={args.resume_from}_EMA'))[-1]
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)
                         
    else:
        ckpt_path = glob(os.path.join(hparams.log.checkpoint_dir,
                         f'*_epoch={args.resume_from}.ckpt'))[-1]
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'])
    print(ckpt_path)
    model.eval()
    model.freeze()
    os.makedirs(hparams.log.test_result_dir, exist_ok=True)

    results=[]
    for i in range(5):
        snr=[]
        base_snr=[]
        lsd=[]
        base_lsd=[]
        t = model.test_dataloader()
        for j, batch in tqdm(enumerate(t)):
            wav, wav_l = batch
            wav=wav.cuda()
            wav_l = wav_l.cuda()
            wav_up = model.sample(wav_l, model.hparams.ddpm.infer_step)
            snr.append(model.snr(wav_up,wav).detach().cpu())
            base_snr.append(model.snr(wav_l, wav).detach().cpu())
            lsd.append(model.lsd(wav_up,wav).detach().cpu())
            base_lsd.append(model.lsd(wav_l, wav).detach().cpu())
            if args.save and i==0:
                swrite(f'{hparams.log.test_result_dir}/test_{j}_up.wav',
                       hparams.audio.sr, wav_up[0].detach().cpu().numpy())
                swrite(f'{hparams.log.test_result_dir}/test_{j}_orig.wav',
                       hparams.audio.sr, wav[0].detach().cpu().numpy())
                swrite(f'{hparams.log.test_result_dir}/test_{j}_linear.wav',
                       hparams.audio.sr, wav_l[0].detach().cpu().numpy())
                swrite(f'{hparams.log.test_result_dir}/test_{j}_down.wav',
                       hparams.audio.sr//hparams.audio.ratio, wav_l[0,::hparams.audio.ratio].detach().cpu().numpy())


        snr = torch.stack(snr, dim =0).mean()
        base_snr = torch.stack(base_snr, dim =0).mean()
        lsd = torch.stack(lsd, dim =0).mean()
        base_lsd = torch.stack(base_lsd, dim =0).mean()
        dict = {
            'snr': snr.item(),
            'base_snr': base_snr.item(),
            'lsd': lsd.item(),
            'base_lsd': base_lsd.item(),
        }
        results.append(torch.stack([snr, base_snr, lsd, base_lsd],dim=0).unsqueeze(-1))
        print(dict)
    results = torch.cat(results,dim=1)
    for i in range(4):
        print(torch.mean(results[i]),torch.std(results[i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume_from', type =int,\
            required = True, help = "Resume Checkpoint epoch number")
    parser.add_argument('-e', '--ema', action = "store_true",\
            required = False, help = "Start from ema checkpoint")
    parser.add_argument('--save', action = "store_true",\
            required = False, help = "Save file")

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = False
    test(args)
