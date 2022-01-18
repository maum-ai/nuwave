# Errata

- Since the evaluation code for SNR contains typo 10 instead of 20, all of the paper's SNR values should be twice.
- There is no difference that our model shows better results than other models on SNR.

- page 4, Table 2

Before

|model|      SingleSpeaker SNR|           MultiSpeaker SNR|
| --- | ----------- | ---------- |
|Linear x2|9.69|11.1|
|U-Net x2|10.3|9.86|
|MU-GAN x2|10.5|12.3|
|NU-Wave x2|**11.1**|**13.2**|
|Linear x3|8.04|8.71|
|U-Net  x3|8.81|10.7|
|MU-GAN x3|9.44|11.7|
|NU-Wave x3|**9.62**|**12.0**|


After 
|model|      SingleSpeaker SNR|           MultiSpeaker SNR|
| --- | ----------- | ---------- |
|Linear x2|19.38|22.2|
|U-Net x2|20.6|19.72|
|MU-GAN x2|21.0|24.6|
|NU-Wave x2|**22.2**|**26.4**|
|Linear x3|16.08|17.42|
|U-Net  x3|17.62|21.4|
|MU-GAN x3|18.88|23.4|
|NU-Wave x3|**19.24**|**24.0**|

- page 4, Section 5

Before
```
Our model improves SNR value by 0.18-0.9 dB from the
best performing baseline, MU-GAN.
```

After
```
Our model improves SNR value by 0.36-1.8 dB from the
best performing baseline, MU-GAN.
```
