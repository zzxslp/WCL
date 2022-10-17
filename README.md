## Weakly Supervised Contrastive Learning for Chest X-Ray Report Generation

This is the pytorch implementation for our paper:

[Weakly Supervised Contrastive Learning for Chest X-Ray Report Generation](https://arxiv.org/abs/2109.12242) 

at Findings of EMNLP-2021.

## Requirements

- `torch>=1.6.0`
- `torchvision>=0.8.0`

## Datasets
We use two datasets (MIMIC-ABN and MIMIC-CXR) in the paper.

For `MIMIC-ABN`, you can download the dataset from [release/mimic_abn](https://drive.google.com/drive/folders/1wokoNJHWh2IN1ywo7t-DKHicZAzwuK2-?usp=sharing) and then put the files in `data/mimic_abn`.

For `MIMIC-CXR`, you can download the dataset from [release/mimic_cxr](https://drive.google.com/drive/folders/1wokoNJHWh2IN1ywo7t-DKHicZAzwuK2-?usp=sharing) and then put the files in `data/mimic_cxr`.

Note: you need to sign user agreements then download x-ray images from [the official website](https://physionet.org/content/mimic-cxr/2.0.0/). 

## Run on MIMIC-ABN

Run `bash run_mimic_abn.sh` to train a model on the MIMIC-ABN data.

## Run on MIMIC-CXR

Run `bash run_mimic_cxr.sh` to train a model on the MIMIC-CXR data.


## Download Models
You can download the models we trained for each dataset from [release/pretrained_models](https://drive.google.com/drive/folders/1wokoNJHWh2IN1ywo7t-DKHicZAzwuK2-?usp=sharing).


## Citation

If you find this repository useful, please cite our paper:

```
@article{yan2021weakly,
  title={Weakly Supervised Contrastive Learning for Chest X-Ray Report Generation},
  author={Yan, An and He, Zexue and Lu, Xing and Du, Jiang and Chang, Eric and Gentili, Amilcare and McAuley, Julian and Hsu, Chun-Nan},
  journal={arXiv preprint arXiv:2109.12242},
  year={2021}
}
```

## Acknowledgments

This project is built on top of [R2Gen](https://github.com/cuhksz-nlp/R2Gen). Thank the authors for their contributions to the community!
