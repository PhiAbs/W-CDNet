# W-CDNet
This repo presents W-CDNet, a neural network for weakly supervised change detection. The model uses a siamese network structure, consisting of two U-Nets with shared weights. 
The core of our model, the *Change Segmentation and Classification* (CSC) module, makes sure that the network learns to generate a change mask even if the model is being trained with weak supervision only.

![W-CDNet model structure](docs/model_all_simplified_cropped.svg)

## Example
Example scripts are provided for training and testing on the AICD dataset. 

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

### Train
Training is split up into "Train" and "Finetune". 

**Train**:
- Go to *Examples/params/aicd_train.py*
  - set a new **run_index** (increase cunter)
  - set **finetune** to False
  - set **fully_supervised** to False
- Go to Examples and run
```
python aicd_train_weakly_supervised.py
```

**Finetune**:
- Go to *Examples/params/aicd_train.py*
  - set **finetune** to True
  - set **fully_supervised** to False
- Go to Examples and run
```
python aicd_train_weakly_supervised.py
```

- The models will be saved under *Examples/models*

### Test
- Go to *Examples/params/aicd_segment_and_classify_and_evaluate.py*
  - set the parameters within the section marked with **adjust these parameters**
- Go to *Examples* and run 
```
chmod +x aicd_run_tests_weakly_supervised.sh
./aicd_run_tests_weakly_supervised.sh
```

- The results will be saved under *Examples/results*


## Image-Level Labels for AICD Dataset
### Dataset
Since the AICD dataset is no longer available on the original website, I uploaded it to google drive. You can download it from [here](https://drive.google.com/file/d/1anlZYIDaZfnFvijg8SfYqt7CvyMDhR_E/view?usp=sharing). <br>
The dataset is also available on kaggle, see [here](https://www.kaggle.com/kmader/aerial-change-detection-in-video-games). <br>

### Labels
You can find the image-level labels in the folder *AICD_image_level_labels*. The file *classes.csv* contains the class IDs and a description for each class. The file *image_level_labels.csv* associates each image (defined by *view* and *scene*) with a class. 


## Cite
[arXiv](https://arxiv.org/abs/2011.03577)

```
@misc{andermatt2020weakly,
      title={A Weakly Supervised Convolutional Network for Change Segmentation and Classification}, 
      author={Philipp Andermatt and Radu Timofte},
      year={2020},
      eprint={2011.03577},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```