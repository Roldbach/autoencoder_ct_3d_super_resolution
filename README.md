# Is Autoencoder Truly Applicable for 3D CT Super-Resolution?

<p align="center">
  <img src="https://github.com/Roldbach/Autoencoder-CT-SISR-3D/blob/main/image/result.pdf" />
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