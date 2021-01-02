[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)

# W-CDNet
This repo presents [W-CDNet](https://arxiv.org/abs/2011.03577), a neural network for weakly supervised change detection. The model uses a siamese network structure, consisting of two U-Nets with shared weights. The core of the model, the *Change Segmentation and Classification* (CSC) module, makes sure that the network learns to generate a change mask even if the model is being trained with weak supervision only.

![W-CDNet model structure](docs/model_all_simplified_cropped.svg)

---
## Demo
Demo scripts are provided for training and testing on the AICD dataset. 
A demo [notebook](Demo/demo.ipynb) is provided which can be run on google Colab. In order to understand how to use the scripts in this repo, please refer to the notebook. Setup steps are shown below.

### Setup
See *requirements.txt* or *requirements_gpu.txt*, depending on whether a GPU is available or not.
Install with:
```
pip install -r requirements.txt
```
or
```
pip install -r requirements_gpu.txt
```

We use the [publicly available keras-implementation for the CRF-RNN layer](https://github.com/sadeepj/crfasrnn_keras).
```
git submodule init
git submodule update
cd crfasrnn_keras/src/cpp
make
``` 

The images for train/val/test can be found [here](https://drive.google.com/file/d/1HLa4xpUZBcK_1__24_QuW3YPFRI_H5rF/view?usp=sharing). Download these images.
```
cd Demo
gdown https://drive.google.com/uc?id=1HLa4xpUZBcK_1__24_QuW3YPFRI_H5rF
unzip AICD_strong_shadows_incl_no_change.zip
```

---
## Image-Level Labels for AICD Dataset

### Dataset
Since the AICD dataset is no longer available on the original website, I uploaded it to google drive. You can download it from [here](https://drive.google.com/file/d/1anlZYIDaZfnFvijg8SfYqt7CvyMDhR_E/view?usp=sharing). <br>
The dataset is also available on kaggle, see [here](https://www.kaggle.com/kmader/aerial-change-detection-in-video-games). <br>

### Labels
You can find the image-level labels in the folder *AICD_image_level_labels*. The file *classes.csv* contains the class IDs and a description for each class. The file *image_level_labels.csv* associates each image (defined by *view* and *scene*) with a class. 

---
## Cite
[arXiv](https://arxiv.org/abs/2011.03577)

```
@inproceedings{andermatt2020weakly,
  title={A Weakly Supervised Convolutional Network for Change Segmentation and Classification},
  author={Andermatt, Philipp and Timofte, Radu},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  year={2020}
}
```