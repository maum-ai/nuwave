from os import path
from torch.utils.data import Dataset, DataLoader
from glob import glob
import torch
from prefetch_generator import BackgroundGenerator
import random
from filters import LowPass

class DataLoader_back(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoader_back, self).__init__(*args, **kwargs)
        if 'num_workers' in kwargs:
            self.num_workers = kwargs['num_workers']
            print('num_workers: ', self.num_workers)
        else:
            self.num_workers = 1

    def __iter__(self):
        return BackgroundGenerator(super().__iter__(),
                                   max_prefetch=self.num_workers // 4)


def create_vctk_dataloader(hparams, cv):
    def collate_fn(batch):
        wav_list = list()
        wav_l_list = list()
        for wav, wav_l in batch:
            wav_list.append(wav)
            wav_l_list.append(wav_l)
        wav_list = torch.stack(wav_list, dim=0).squeeze(1)
        wav_l_list = torch.stack(wav_l_list, dim=0).squeeze(1)

        return wav_list, wav_l_list

    if cv==0:
        return DataLoader_back(dataset=VCTKMultiSpkDataset(hparams, cv),
                               batch_size=hparams.train.batch_size,
                               shuffle=True,
                               num_workers=hparams.train.num_workers,
                               collate_fn=collate_fn,
                               pin_memory=True,
                               drop_last=True,
                               sampler=None)
    else:
        return DataLoader_back(dataset=VCTKMultiSpkDataset(hparams, cv),
                               collate_fn=collate_fn,
                               batch_size=hparams.train.batch_size if cv==1 else 1,
                               drop_last=True if cv==1 else False,
                               shuffle=False,
                               num_workers=hparams.train.num_workers)


class VCTKMultiSpkDataset(Dataset):
    def __init__(self, hparams, cv=0):  #cv 0: train, 1: val, 2: test
        def _get_datalist(folder, file_format, spk_list, cv):
            _dl = []
            len_spk_list = len(spk_list)
            s=0
            print(f'full speakers {len_spk_list}')
            for i, spk in enumerate(spk_list):
                if cv==0:
                    if not(i<int(len_spk_list*self.cv_ratio[0])): continue
                elif cv==1:
                    if not(int(len_spk_list*self.cv_ratio[0])<=i and
                            i<=int(len_spk_list*(self.cv_ratio[0]+self.cv_ratio[1]) )):
                        continue
                else:
                    if not(int(len_spk_list*self.cv_ratio[0])<=i and
                            i<=int(len_spk_list*(self.cv_ratio[0]+self.cv_ratio[1]) )):
                        continue
                _full_spk_dl = sorted(glob(path.join(spk, file_format)))
                _len = len(_full_spk_dl)
                if (_len == 0): continue
                s+=1    
                _dl.extend(_full_spk_dl)
            
            print(cv, s)
            return _dl

        def _get_spk(folder):
            return sorted(glob(path.join(folder, '*')))#[1:])
        
        self.hparams = hparams
        self.cv = cv
        self.cv_ratio = eval(hparams.data.cv_ratio)
        self.directory = hparams.data.dir
        self.dataformat = hparams.data.format
        self.data_list = _get_datalist(self.directory, self.dataformat,
                                       _get_spk(self.directory), self.cv)

        self.filter_ratio = [1./hparams.audio.ratio]
        self.lowpass = LowPass(hparams.audio.nfft,
                               hparams.audio.hop,
                               ratio=self.filter_ratio)
        self.upsample = torch.nn.Upsample(scale_factor=hparams.audio.ratio,
                                          mode ='linear',
                                          align_corners = False)
        assert len(self.data_list) != 0, "no data found"

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        wav = torch.load(self.data_list[index])
        wav /= wav.abs().max()
        if wav.shape[0] < self.hparams.audio.length:
            padl = self.hparams.audio.length - wav.shape[0]
            r = random.randint(0, padl) if self.cv<2 else padl//2
            wav = torch.nn.functional.pad(wav, (r, padl-r), 'constant', 0)
        else:
            start = random.randint(0, wav.shape[0] - self.hparams.audio.length)
            wav = wav[start:start+self.hparams.audio.length] if self.cv<2 \
                    else wav[:len(wav)-len(wav)%self.hparams.audio.ratio]
        wav *= random.random()/2+0.5 if self.cv<2 else 1

        wav_l = self.lowpass(wav, 0)
        wav_l = wav_l[0,::self.hparams.audio.ratio].view(1,1,-1)
        #or
        #wav_l = rosa.resample(wav, hparams.audio.sr, hparams.audio.sr//hparams.audio.ratio)
        wav_l = self.upsample(wav_l).view(1,-1)
        return wav, wav_l


class VCTKSingleSpkDataset(Dataset):
    def __init__(self, hparams, cv=0):  # cv 0: train, 1: val, 2: test
        def _get_datalist(folder, file_format, cv):
            _dl = []
            audio_list = sorted(glob(path.join(folder, file_format)))
            len_audio_list = len(audio_list)
            s=0
            print(f'full audios {len_audio_list}')
            for i, audio in enumerate(audio_list):
                if cv == 0:
                    if not (i < int(len_audio_list * self.cv_ratio[0])): continue
                elif cv == 1:
                    if not (int(len_audio_list * self.cv_ratio[0]) <= i and
                            i <= int(len_audio_list * (self.cv_ratio[0] + self.cv_ratio[1]))):
                        continue
                else:
                    if not (int(len_audio_list * self.cv_ratio[0]) <= i and
                            i <= int(len_audio_list * (self.cv_ratio[0] + self.cv_ratio[1]))):
                        continue
                s+=1
                _dl.append(audio)

            print(cv, s)
            return _dl

        self.hparams = hparams
        self.cv = cv
        self.cv_ratio = eval(hparams.data.cv_ratio)
        self.directory = hparams.data.dir
        self.dataformat = hparams.data.format
        self.data_list = _get_datalist(self.directory, self.dataformat, self.cv)

        self.filter_ratio = [1./hparams.audio.ratio]
        self.lowpass = LowPass(hparams.audio.nfft,
                               hparams.audio.hop,
                               ratio=self.filter_ratio)
        self.upsample = torch.nn.Upsample(scale_factor=hparams.audio.ratio,
                                          mode ='linear',
                                          align_corners = False)
        assert len(self.data_list) != 0, "no data found"

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        wav = torch.load(self.data_list[index])
        wav /= wav.abs().max()
        if wav.shape[0] < self.hparams.audio.length:
            padl = self.hparams.audio.length - wav.shape[0]
            r = random.randint(0, padl) if self.cv<2 else padl//2
            wav = torch.nn.functional.pad(wav, (r, padl-r), 'constant', 0)
        else:
            start = random.randint(0, wav.shape[0] - self.hparams.audio.length)
            wav = wav[start:start+self.hparams.audio.length] if self.cv<2 \
                    else wav[:len(wav)-len(wav)%self.hparams.audio.ratio]
        wav *= random.random()/2+0.5 if self.cv<2 else 1

        wav_l = self.lowpass(wav, 0)
        wav_l = wav_l[0,::self.hparams.audio.ratio].view(1,1,-1)
        #or
        #wav_l = rosa.resample(wav, hparams.audio.sr, hparams.audio.sr//hparams.audio.ratio)
        wav_l = self.upsample(wav_l).view(1,-1)
        return wav, wav_l