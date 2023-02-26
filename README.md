# Is Autoencoder Truly Applicable for 3D CT Super-Resolution?

<p align='center'>
  <img src='./image/result.png'/>
</p> 

> **Is Autoencoder Truly Applicable for 3D CT Super-Resolution?**<br>
> Weixun Luo, Xiaodan Xing, Guang Yang<br>
> ISBI 2023
> 
> **Abstract**: <br>
> Featured by a bottleneck structure, autoencoder (AE) and its variants have
> been largely applied in various medical image analysis tasks, such as
> segmentation, reconstruction and de-noising. Despite of their promising
> performances in aforementioned > tasks, in this paper, we claim that AE models
> are not applicable to single image super-resolution (SISR) for 3D CT data. Our
> hypothesis is that the bottleneck architecture that resizes feature maps in AE
> models degrades the details of input images, thus can sabotage the performance
> of superresolution. Although U-Net proposed skip connections that merge
> information from different levels, we claim that the degrading impact of
> feature resizing operations could hardly be removed by skip connections. By
> conducting large-scale ablation experiments and comparing the performance
> between models with and without the bottleneck design on a public CT lung
> dataset, we have discovered that AE models, including U-Net, have failed to
> achieve a compatible SISR result (p < 0.05 by Student’s t-test) compared to
> the baseline model. Our work is the first comparative study investigating the
> suitability of AE architecture for 3D CT SISR tasks and brings a rationale for
> researchers to re-think the choice of model architectures especially for 3D CT
> SISR tasks.


## Description
This repository contains the official PyTorch Implementation of paper: Is
Autoencoder Truly Applicable for 3D CT Super-Resolution? We have provided our
full training and inference codes and pre-trained models as well.


## Get Started
### Install Virtual Environment
To install the environment, please clone the repository and run the following
commands:

```shell script
conda env create -f environment.yml
conda activate 3DSuperResolution
```

### Dataset
To use a customised dataset, please provide a dataset loading file as **.csv**
and modify `DATASET_LOADING_FILE_PATH` in
[dataset_configuration.py]('./configuration/dataset_configuration.py'). This
file should contains the following columns:

- `path`: A **str** that specifies the path of a 3D image
- `tag`: A **str** that specifies the subset, one of `train`/`validation`/`test`.


We have also provided a [sample loading file]('./dataset/sample_loading_file.csv')
as demonstration.


## Train models on customised datasets
To train models on customised datasets:
- Prepare the dataset by following the instructions above.
- Go to [train.py]('./train.py') and specify the following settings:

    - model_name <br>
      A **str** that specifies the name of the model used in the experiment,
      choose from **PlainCNN / AE_Maxpool / AE_Conv / UNet**.
    - upsample_name <br> 
      A **str** that specifies the name of the upsampling method used in
      low-resolution data generation, choose from
      **trilinear_interpolation / same_insertion**.
    - scale_factor <br>
      An **int** that specifies the scale_factor of downsampling/upsampling in
      the z-axis , **2 / 4 / 8** used in the paper.
    - weight_path <br>
      A **str** that specifies the file path to store the model weight ended
      with **.pth**.
    - record_path <br>
      A **str** that specifies the file path to store the record of experiment
      ended with **.csv**.
    - require_loading <br>
      A **bool** that specifies whether to continue training from the last
      experiment, otherwise restart training from the scratch.
    - window <br>
      A **tuple[float|None, float|None]** that specifies the range of pixel
      values interested.
    - patch_size <br>
      An **int** that specifies the size of cubic patches.
    - epoch <br>
     An **int** that specifies the total number of iterations during training.
    - batch_size <br>
      An **int** that specifies the number of data in one batch.
    - learning_rate <br>
      A **float** that specifies the step size in gradient updating.


### Model

<p align='center'>
  <img src='./image/model_all.png'/>
</p> 

We have provided the following models:

  - `PlainCNN`: A cascade of (Conv3D + LeakyReLU) blocks + global residual
    learning
  - `AE_Maxpool`: Use PlainCNN as baseline and Maxpooling as downsampling
    method.
  - `AE_Conv`: Use PlainCNN as baseline and strided Conv3D as downsampling
    method.
  - `UNet`: A simplified 3D UNet implementation for fair comparisons.

