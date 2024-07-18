# 

This repository is for providing code for CVPR 2024 submission.

## Installation

### Prerequisite
Following libraries are required.  
Please follow installation guide below to avoid dependency and path issues.
- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)
- [Matterport3D Dataset](https://niessner.github.io/Matterport/)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MiDaS](https://github.com/isl-org/MiDaS)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)


#### [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
We tested with `habitat-sim` version `v0.2.1`.  
The coordinate configuration seems to be changed in upper version. So, please use `v0.2.1`.

```bash
# Make conda env
conda create -n habitat python=3.8 cmake=3.14.0  # python version 3.8 is used for habitat-sim v0.2.1
conda activate habitat
# Install with pybullet & GUI
conda install habitat-sim=0.2.1 withbullet -c conda-forge -c aihabitat
```

#### [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)
We tested with `habitat-lab` version `v0.2.1`.

```bash
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout v0.2.1
pip install -e .
```

#### [PyTorch](https://pytorch.org/get-started/previous-versions/)
We tested with `torch` version `1.10.1`.  
Please install with `pip` to avoid dependency issue with `PaddleOCR`.

```bash
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

#### [MMDetection](https://mmdetection.readthedocs.io/en/latest/get_started.html)
```bash
pip install -U openmim
mim install mmengine mmcv==2.0.1 mmdet==3.1.0

cd /your/path/to/code
mkdir model_weights
mim download mmdet --config rtmdet_m_8xb32-300e_coco --dest ./model_weights/
```

#### [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/quickstart_en.md)
We tested with `paddlepaddle` version `2.5.0` and `paddleocr` version `2.6.1.3`.

```bash
python -m pip install paddlepaddle-gpu==2.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install paddleocr==2.6.1.3
```

#### Other dependencies
```bash
pip install -r requirements.txt
```

#### Matterport3D Dataset
[Matterport3D dataset](https://niessner.github.io/Matterport/) is required for test in various indoor environments.  
To get this dataset, please visit [here](https://niessner.github.io/Matterport/) and submit Terms of Use agreement.  
You will get `download_mp.py` file after you submit the form.

```bash
# Download dataset. Need python 2.7. Dataset is about 15GB.
python2 download_mp.py --task habitat -o /your/path/to/download

# Make dataset directory.
# If you don't want to store dataset in this repo directory, fix SCENE_DIRECTORY in config file
cd /your/path/to/code  # clone of this repository
mkdir Matterport3D

# Unzip 
unzip /your/path/to/download/v1/tasks/mp3d_habitat.zip -d ./Matterport3D
```


## Run test in simulation
You can generate floor plans yourself with just running `generate_grid_map.py`.  
However, it takes quite a long time, so we've already generated map files in `data` directory.

```bash
export MAGNUM_LOG=quiet GLOG_minloglevel=2  # To mute logs from Habitat-env
python run_localization_test.py
```

## Run test in Real world
Directory `query_imgs` contains sample pictures.  
File `free_area_dense.csv` contains navigable locations in the human readable map. (Over 3 milion points)  
File `store_info_eng.csv` contains store names and locations extracted from the human readable map.  

The map image itself could not be uploaded due to the double blind policy.  

Due to the personal information and double blind issue, limited number of samples are provided for real world experiment.  
The experiment in the paper includes more pictures.

```bash
python run_localization_realworld.py
```
