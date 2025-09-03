# PBFG: A New Physically-Based Dataset and Removal of Lens Flares and Glares (ICCV 2025)<br><sub>Official Implementation</sub><br>

This repository contains the official dateset and code in the following paper:

[PBFG: A New Physically-Based Dataset and Removal of Lens Flares and Glares](http://115.145.173.100/pub/2025-zhu-iccv-pbfg).

[Jie Zhu¹](https://cg.skku.edu/ppl/), [Sungkil Lee¹](https://cg.skku.edu/slee/)

¹Sungkyunkwan University

*International Conference on Computer Vision （ICCV） 2025*

## Overview
Flare and glare are common nighttime artifacts that degrade image quality and hinder computer vision tasks. Existing synthetic datasets lack physical realism and diversity, while deep learning-based removal methods struggle in complex scenes, posing significant challenges. To address these issues, we introduce the high-quality annotated Physically-Based Flare and Glare (PBFG) dataset and a Flare and Glare Removal Network (FGRNet). PBFG comprises 2,600 flares and 4,000 glares using our computational rendering scheme with diverse lens systems and optical configurations. Our advanced streak synthesis enhances template fidelity and improves streak removal accuracy. FGRNet leverages spatial-frequency features for comprehensive local and global feature extraction. It introduces a Spatial-Frequency Enhanced Module with a Spatial Reconstruction Unit and a Frequency-Enhanced Unit to extract multi-scale spatial information and enhance frequency representation. This design effectively removes complex artifacts, including large-area glares, diverse flares, and multiple or off-screen-induced streaks. Additionally, a histogram-matching module ensures stylistic and visual consistency with ground truth. Extensive experiments confirm that PBFG accurately replicates real-world patterns, and FGRNet outperforms state-of-the-art methods both quantitatively and qualitatively, resulting in significant gains of PSNRs (up to 2.3 dB and 3.14 dB in an image and its glare regions, respectively).

## Dataset Download

We use Git Large File Storage (LFS) to manage the PBFG dataset. So please use LFS to download the dataset.
 
### PBFG Dataset

The PBFG Dataset is in `dataset/`.

### PBStar Dataset

The PBStar Dataset is in `dataset/PBStar.zip`.
 
## Code

### Installation

1. Clone the repo

    ```bash
    git clone https://github.com/cgskku/pbfg.git
    ```

1. Install dependent packages

    ```bash
    cd pbfg
    pip install -r requirements.txt
    ```

1. Install pbfg<br>
    Please run the following commands in the **pbfg root path** to install pbfg:<br>

    ```bash
    python setup.py develop
    ```

### Pre-trained Model

Our pretrained checkpoint is in `code/PBFG/experiments/checkpoint.zip`.


### Test Data

The test dataset is in `dataset/test_data.zip`.


### Inference Code
To estimate the flare/glare-free images with our checkpoint pretrained on PBFG, you can run the `test.py` by using:
```
python test.py --gt dataset/test_data/real/gt  --input dataset/test_data/real/input   --output result/test_real/pbfg/  --model_path experiments/checkpoint/net_g_last.pth   --flare7kpp
```

### Evaluation Code
To calculate different metrics with our pretrained model, you can run the `evaluate.py` by using:
```
python evaluate.py --input result/test_real/pbfg/blend/  --gt  dataset/test_data/real/gt/ --mask  dataset/test_data/real/mask/
```

### Training model

**Training with single GPU**

To train a model with your own data/model, you can edit the `options/uformer_flare7kpp_baseline_option.yml` and run the following codes. You can also add `--debug` command to start the debug mode:

```
python basicsr/train.py -opt options/uformer_flare7kpp_baseline_option.yml
```

**Training with multiple GPU**

You can run the following command for the multiple GPU tranining:

```
CUDA_VISIBLE_DEVICES=0,1 bash scripts/dist_train.sh 2 options/uformer_flare7kpp_baseline_option.yml
```

### PBFG structure

```
├── PBFG
    ├── flare
    ├── glare
         ├── compound_glare
         ├── glow
         ├── light_source
         ├── shimmer
         ├── starburst
         ├── streak
├── PBStar
```

### License

This project is licensed under CC BY-NC-SA 4.0. Redistribution and use of the dataset and code for non-commercial purposes should follow this license.

### Citation

If you find this work useful, please cite:

```
@inproceedings{zhu2025pbfg,
  title={PBFG: A New Physically-Based Dataset and Removal of Lens Flares and Glares},
  author={Zhu, Jie and Lee, Sungkil},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

### Acknowledgement
This repository is based on the [Flare7K](https://github.com/ykdai/Flare7K). Thanks for their awesome work.
