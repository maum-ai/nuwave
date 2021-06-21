from lightning_model import NuWave
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf as OC
import os
import argparse
import datetime
from glob import glob
import torch

def test(args):
    hparams = OC.load('hparameter.yaml')
    hparams.save = args.save or False
    model = NuWave(hparams, False)
    
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

    trainer = Trainer(
        gpus=1,
        amp_backend='apex',  #
        amp_level='O2',  #
        progress_bar_refresh_rate=4,
        )
    #for i in range(5):
    trainer.test(model, ckpt_path = 'None')


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
