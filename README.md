# Calisthenics Skills Temporal Video Segmentation

This repository hosts the keypoints dataset and some additional codes related to the paper:  
`
Finocchiaro, A.; Farinella, G. and Furnari, A. (2024). Calisthenics Skills Temporal Video Segmentation.  In Proceedings of the 19th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 2: VISAPP, ISBN 978-989-758-679-8, ISSN 2184-4321, pages 182-190.
`

Other works involved:

`
[JVCI 2018] A. Furnari, S. Battiato, G. M. Farinella, Personal-Location-Based Temporal Segmentation of Egocentric Video for Lifelogging Applications . Journal of Visual Communication and Image Representation , 52 , pp. 1-12
`
## Repository Overview

### Data

The `/data` directory you can find the `keypoints_dataset`, presented and discussed in the paper. 

### Source Code

The `/src` directory contains the source code organized into three subfolders:

- inference_scripts

- **codec.py**: This script encodes and decodes labels, essential for the inference process.
- **inference.py**: The main inference script. It tests the entire pipeline, taking a video's relative path as an argument and displaying corresponding segments with their times.
- **openpose_script.py**: This script extracts keypoints from a video and is utilized within the inference script.

- model

- **mlp.py**: This script defines the architecture of the multilayer perceptron (MLP), encompassing both training and testing phases.

- temporal_segmentation

- **furnari2018.py**: Implementation of the Probabilistic algorithm.
- **heuristic.py**: Implementation of the Heuristic algorithm.

## Requirements 

To execute inference on a new video, begin by installing the required libraries listed in the requirements.txt file, running the following command. 

```bash
pip3 install -r requirements.txt
```

Then, you will need to install OpenPose in your computer, all the steps for its installation are listed in the following link: 

<a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md">OpenPose installation guide</a>

