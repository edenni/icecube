# IceCube


## Installation

python==3.7.16


### Local

Build graphnet from source https://github.com/graphnet-team/graphnet#gear--install

```
pip install -e .
```


Install dependencies match your pytorch version. i.e 

```
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html
```

See [this](https://github.com/rusty1s/pytorch_cluster#binaries) for detail.


### Kaggle kernel

Since the internet is unavailble when you submit the kernel, so you have to upload the wheels in advance.
Refer the [official baseline](https://www.kaggle.com/code/rasmusrse/graphnet-baseline-submission) for detail.