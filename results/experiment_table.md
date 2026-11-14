| Config | Varied factor | Seeds | mAP50 (mean, range) | mAP50-95 (mean, range) | Train min/run |
|---|---|---|---|---|---|
| baseline | reference (imgsz 640, default aug, 12 epochs) | 2 | 0.6105 (0.6094-0.6116) | 0.3791 (0.3783-0.3799) | 12.3 |
| strong_aug | augmentation strength (mixup .2, scale .9, hsv_s .9, hsv_v .6, rot 5deg) | 2 | 0.5855 (0.5847-0.5863) | 0.3547 (0.3546-0.3548) | 11.9 |
| hires | input resolution (imgsz 960) | 2 | 0.7022 (0.6999-0.7045) | 0.4574 (0.4559-0.4590) | 23.1 |
| degraded | training length (1 epoch) | 1 | 0.3271 (0.3271-0.3271) | 0.1700 (0.1700-0.1700) | 2.5 |
