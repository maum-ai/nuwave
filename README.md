# NU-Wave &mdash; Official PyTorch Implementation

**NU-Wave: A Diffusion Probabilistic Model for Neural Audio Upsampling**<br>
Junhyeok Lee, Seungu Han @ [MINDsLab Inc.](https://github.com/mindslab-ai), SNU

Paper(arXiv): https://arxiv.org/abs/2104.02321 (Accepted to INTERSPEECH 2021)<br>
Audio Samples: https://mindslab-ai.github.io/nuwave<br>

Official Pytorch+[Lightning](https://github.com/PyTorchLightning/pytorch-lightning) Implementation for NU-Wave.<br>

Update: **CODE RELEASED!** README is still updating.<br>
TODO: How to preprocessing/ training/ evaluation

## Preprocessing
TODO

## Training
TODO
run `trainer.py`

## Evaludation
TODO
run `for_test.py` or `test.py`

## Repository Structure
```
.
├── Dockerfile				
├── dataloader.py			# Dataloader for train/val(=test)
├── filters.py				# Filter implementation
├── test.py					# Test with lightning_loop.
├── for_test.py				# Test with for_loop. Recommended due to device dependency of lightning
├── hparameter.yaml			# Config
├── lightning_model.py		# NU-Wave implementation. DDPM is based on ivanvok's WaveGrad implementation
├── model.py				# NU-Wave model based on lmnt-com's DiffWave implementation
├── requirement.txt         # requirement libraries
├── sampling.py             # Sampling a file
├── trainer.py              # Lightning trainer
├── README.md
├── utils
│  ├── stft.py              # STFT layer
│  ├── tblogger.py          # Tensorboard Logger for lightning
│  └── wav2pt.py            # Preprocessing
└── docs                    # For github.io
    └─ ...
```

## Requirements
Pytorch >=1.7.0 for nn.SiLU(swish)
Pytorch-Lightning==1.1.6
The requirements are highlighted in [requirements.txt](./requirements.txt).
We also provide docker setup [Dockerfile](./Dockerfile).

## References
This implementation uses code from following repositories:
- [J.Ho's official DDPM implementation](https://github.com/hojonathanho/diffusion)
- [lucidrain's DDPM pytorch implementation](https://github.com/lucidrains/denoising-diffusion-pytorch)
- [ivanvok's WaveGrad pytorch implementation](https://github.com/ivanvovk/WaveGrad)
- [lmnt-com's DiffWave pytorch implementation](https://github.com/lmnt-com/diffwave)

This README and the webpage for the audio samples are inspired by:
- [Tips for Publishing Research Code](https://github.com/paperswithcode/releasing-research-code)
- [Audio samples webpage of DCA](https://google.github.io/tacotron/publications/location_relative_attention/)
- [Cotatron](https://github.com/mindslab-ai/cotatron/)
- [Audio samples wabpage of WaveGrad](https://wavegrad.github.io)

The audio samples on our [webpage](https://mindslab-ai.github.io/nuwave/) are partially derived from:
- [VCTK](https://datashare.ed.ac.uk/handle/10283/3443): 46 hours of English speech from 108 speakers.

## Citation & Contact
If this repository useful for your research, please consider citing!
Bibtex will be updated after INTERSPEECH 2021 conference.
```bib
@article{lee2021nuwave,
  title={NU-Wave: A Diffusion Probabilistic Model for Neural Audio Upsampling},
  author={Lee, Junhyeok and Han, Seungu},
  journal={arXiv preprint arXiv:2104.02321},
  year={2021}
}
```
If you have a question or any kind of inquiries, please contact Junhyeok Lee at [jun3518@mindslab.ai](mailto:jun3518@mindslab.ai)
