# IceCube

## Solution

Model: 2 * LSTM + 2 * GRU, regression -> classification with shifting

Features:
- using `auxiliary` and other feature to create `priority`
-  picking up important samples with window
  
See those data processing notebook for detail.   

## Installation

python==3.7.16

### Kaggle

Since the internet is unavailble when you submit the kernel, so you have to upload the wheels in advance.
Refer the [official baseline](https://www.kaggle.com/code/rasmusrse/graphnet-baseline-submission) for detail.
Other libs TBD

### Local

#### GraphNeT

```
cd src/graphnet && pip install -e .
```


#### Dependencies of torch-geometric

Install following libraries match your torch and cuda version.

```
pip install torch-cluster torch-scatter torch-sparse torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html
```

See [this](https://github.com/rusty1s/pytorch_cluster#binaries) for detail.
