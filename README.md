# NeRF_from_scratch
Nerf with minimalistic implementation of NeRF from paper [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)



If you are new to the field of NeRF, I recommend watching [this video](https://youtu.be/JuH79E8rdKc?si=52Jg5jvaiNUKU_jN) and reading the original [NeRF paper](https://arxiv.org/abs/2003.08934).
If you are new to <b>'Positional Encoding'</b> which is one of the key inventions that made NeRFs and transformers possible, checkout my other repo [Fourier_feature_positional_encoding](https://github.com/HrushikeshBudhale/Fourier_feature_positional_encoding).

The code is written in a way to make it easier for learners to understand the concepts and add new changes and models on top of it.

## 'main' branch

This branch implements stratified sampling with single network.

<p align="center">
  <img src="https://github.com/HrushikeshBudhale/NeRF_from_scratch/blob/main/results/train_output.gif?raw=true" alt="NeRF Results">
</p>
<p align="center">This small model can achieve a PSNR score of 25.0.</p>
<p align="center">
  <img width="400" src="https://github.com/HrushikeshBudhale/NeRF_from_scratch/blob/main/results/psnr_score.png?raw=true" alt="NeRF Results">
</p>

Novel view synthesis results:

<p align="center">
  <img src="https://github.com/HrushikeshBudhale/NeRF_from_scratch/blob/main/results/rgb_output.gif?raw=true" alt="NeRF Results">
  <img src="https://github.com/HrushikeshBudhale/NeRF_from_scratch/blob/main/results/depth_output.gif?raw=true" alt="NeRF Results">
</p>

## Branch 'nerf_with_hierarchical_sampling'

This branch implements hierarchical sampling that uses additional samples making the reconstruction more accurate.

*(Do checkout the difference between the two branches to understand the little change needed to implement hierarchical sampling.)*

![NeRF Results](github.comtrain_output_hierarchical.gif)
You can find the data required to train the model here on [gdrive](https://drive.google.com/drive/folders/1dDeBOoNJUWqrQI1zkl_qyk0kF2MZMu_z?usp=sharing).

<p align="center">
<img src="https://github.com/HrushikeshBudhale/NeRF_from_scratch/blob/nerf_with_hierarchical_sampling/results/rgb_output.gif?raw=true" alt="NeRF Results">
<img src="https://github.com/HrushikeshBudhale/NeRF_from_scratch/blob/nerf_with_hierarchical_sampling/results/depth_output.gif?raw=true" alt="NeRF Results">
</p>

With hierarchical sampling recon can achieve higher PSNR value. (from previous 25.0 to 28.6).


<p align="center">
  <img src="https://github.com/HrushikeshBudhale/NeRF_from_scratch/blob/nerf_with_hierarchical_sampling/results/train_output.gif?raw=true" alt="NeRF Results">
</p>

<p align="center">
  <img width="400" src="https://github.com/HrushikeshBudhale/NeRF_from_scratch/blob/nerf_with_hierarchical_sampling/results/psnr_score.png?raw=true" alt="NeRF Results">
</p>

## Installation

1. Create a conda environment with python 3.10.

    ```bash
    conda create -n nerf python=3.10
    conda activate nerf
    ```

2. Install supported version of pytorch.

3. Install the dependencies.

    ```bash
    pip install -r requirements.txt
    ```

4. Download the scene data from [gdrive](https://drive.google.com/drive/folders/1dDeBOoNJUWqrQI1zkl_qyk0kF2MZMu_z?usp=sharing). (or use your own data)

5. Update the config file with the path to the data.

6. Run the code.

    ```bash
    python train_nerf.py
    python test_nerf.py
    ```

NeRFs and Gaussian Splatting have revolutionized the field of 3D reconstruction and view synthesis. Instant Neural Graphics Primitives ([InstantNGP](https://github.com/NVlabs/instant-ngp)) by NVIDIA took NeRFs to the next level by making it much faster and more accurate.

If you are interested in learning NGP checkout my other repo [NGP_from_scratch](https://github.com/HrushikeshBudhale/NGP_from_scratch) that builds on top of this code and achieves even better results.

## Acknowledgements

- [NeRF paper](https://arxiv.org/abs/2003.08934)
- [tyny_Nerf](https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb)
- [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)
