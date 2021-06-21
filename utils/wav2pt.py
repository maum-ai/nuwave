import torch
import librosa as rosa
from omegaconf import OmegaConf as OC
import os
from glob import glob
from tqdm import tqdm
import multiprocessing as mp

def wav2pt(wav):
    y,_ = rosa.load(wav, sr = hparams.audio.sr, mono = True)
    y,_ = rosa.effects.trim(y, 15)
    pt_name = os.path.splitext(wav)[0]+'.pt'
    pt = torch.tensor(y)
    torch.save(pt ,pt_name)
    del y, pt 
    return

if __name__=='__main__':
    hparams = OC.load('hparameter.yaml')
    dir = hparams.data.dir
    wavs = glob(os.path.join(dir, '*/*.flac'))
    pool = mp.Pool(processes = hparams.train.num_workers)
    with tqdm(total = len(wavs)) as pbar:
        for _ in tqdm(pool.imap_unordered(wav2pt, wavs)):
            pbar.update()

            
