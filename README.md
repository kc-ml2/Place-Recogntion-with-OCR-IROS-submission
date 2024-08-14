# 

This repository is for providing code for CVPR 2024 submission.

## Installation

### Prerequisite
Following libraries are required.  
Please follow installation guide below to avoid dependency and path issues.
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [MiDaS](https://github.com/isl-org/MiDaS)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)


#### [PyTorch](https://pytorch.org/get-started/previous-versions/)
We tested with `torch` version `2.3.1`.  
Please install with `pip` to avoid dependency issue with `PaddleOCR`.

```bash
pip install torch torchvision
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

## Run test
Directory `query_imgs` contains sample pictures.  
File `free_area_dense.csv` contains navigable locations in the human readable map. (Over 3 milion points)  
File `store_info_eng.csv` contains store names and locations extracted from the human readable map.  

Due to the personal information, limited number of samples are provided for real world experiment.  
The experiment in the paper includes more pictures.

```bash
python run_localization_realworld.py

# Visualixation
python run_localization_realworld.py --visualize
```
