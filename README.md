# Monocular Depth Estimation Using MiDaS

## How To Build
1. Download and install [Anaconda](https://anaconda.org/)
2. Clone this repository locally and navigate to its folder.
3. Create and activate the conda environment
```
conda env create -f environment.yaml
conda activate midas-py310
```
5. Run the script `python main.py`

## Overview
This project was created from [this](https://medium.com/artificialis/getting-started-with-depth-estimation-using-midas-and-python-d0119bfe1159) resource and served as an introductory resource into depth estimation. This script uses a pretrained depth estimation model named [MiDaS](https://github.com/isl-org/MiDaS). The dependnacies are managed by [Anaconda](https://anaconda.org/). This script opens up two windows, one shows the original camera input and the other shows a heatmap of the output of MiDaS. The script calculates distance as the inverse of the depth multiplied by some scale factor. For mostly static scenes, the scale factor can be tuned to get an approximate distance estimation in units such as meters. See why this does not work for dynamic scenes in the [shortcommings section](## Shortcommings).

## Shortcommings
The depth information given from the MiDaS model is relative. This means that in a dynamic scene, as objects move around the same depth value can mean different things. Absolute depth estimation was not accomplished in this project because of time constraints. In order to implement such absolute depth estimation one might look at equations 1-4 of the original [MiDaS paper](https://arxiv.org/pdf/1907.01341v3).
