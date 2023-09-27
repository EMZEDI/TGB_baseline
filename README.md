# Temporal Graph Neural Networks Fail to Capture Global Temporal Dynamics

## Introduction
This repository contains the codebase supporting the findings of the paper "Temporal Graph Neural Networks Fail to Capture Global Temporal Dynamics." The repository focuses on exploring the limitations of temporal graph neural networks in capturing global temporal dynamics. This work utilizes code from the [Original Challenge Benchmark Repo](https://github.com/shenyangHuang/TGB).


## Generating Recently Popular Negative Samples
To bring evaluation results closer to real-world usefulness, we propose an improved evaluation method. This involves generating negative samples based on the popularity of items. Run the following script to generate these samples:

```bash
python tgb/datasets/dataset_scripts/popularity_neg_generator.py
```

The generated negative samples will be saved in `output/popular_neg_samples/{dataset_name}`.

## Model Evaluation

### Evaluating EdgeBank
Run the following command to evaluate EdgeBank:

```bash
python examples/linkproppred/edgebank.py --data <your-data-here> --mem_mode unlimited
```

### Training and Evaluating TGN
- **Naive Negative Sampling**:
```bash
python examples/linkproppred/tgn.py --data <your-data-here> --sampling-strategy naive
```

- **Temporal Popularity Negative Sampling**:
```bash
python examples/linkproppred/tgn.py --data <your-data-here> --random-ratio 0.1 --sampling-strategy popularity
```
`random-ratio` is the ratio of naive negative samples to temporal popularity negative samples.

### Training and Evaluating DyRep
- **Naive Negative Sampling**:
```bash
python examples/linkproppred/dyrep.py --data <your-data-here> --sampling-strategy naive
```

- **Temporal Popularity Negative Sampling**:
```bash
python examples/linkproppred/dyrep.py --data <your-data-here> --random-ratio 0.1 --sampling-strategy popularity
```

Replace `<your-data-here>` with the dataset you intend to use (e.g., tgbl-coin).

## Temporal Popularity Baseline
To run the temporal popularity baseline for dynamic link property prediction, execute:

```bash
python examples/linkproppred/popularity_baseline.py
```

## Analysis of Global Temporal Dynamics in Datasets
This repository also includes notebook that present an analysis on measuring global temporal dynamics in temporal graph datasets. The analysis is based on the findings described in our paper and aims to quantify how much information recent global destination node popularity provides for future edges in a temporal graph dataset.

To explore this analysis, refer to the notebook `notebooks/tgb_nonstationarity.ipynb`.



---------------------------------------------


### Links and Datasets

The project website can be found [here](https://tgb.complexdatalab.com/).

The API documentations can be found [here](https://shenyanghuang.github.io/TGB/).

all dataset download links can be found at [info.py](https://github.com/shenyangHuang/TGB/blob/main/tgb/utils/info.py)

TGB dataloader will also automatically download the dataset as well as the negative samples for the link property prediction datasets.

if website is unaccessible, please use [this link](https://tgb-website.pages.dev/) instead.


### Install dependency
Our implementation works with python >= 3.9 and can be installed as follows

1. set up virtual environment (conda should work as well)
```
python -m venv ~/tgb_env/
source ~/tgb_env/bin/activate
```

2. install external packages
```
pip install pandas==1.5.3
pip install matplotlib==3.7.1
pip install clint==0.5.1
```

install Pytorch and PyG dependencies (needed to run the examples)
```
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu117
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

3. install local dependencies under root directory `/TGB`
```
pip install -e .
```


### full dependency list
Our implementation works with python >= 3.9 and has the following dependencies
```
pytorch == 2.0.0
torch-geometric == 2.3.0
torch-scatter==2.1.1
torch-sparse==0.6.17
torch-spline-conv==1.2.2
pandas==1.5.3
clint==0.5.1
```
