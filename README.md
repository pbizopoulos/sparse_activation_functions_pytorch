[![citation](http://img.shields.io/badge/Citation-0091FF.svg)](https://scholar.google.com/scholar?q=Sparsely%20Activated%20Networks.%20arXiv%202020)
[![arXiv](http://img.shields.io/badge/cs.LG-arXiv%3A1907.06592-B31B1B.svg)](https://arxiv.org/abs/1907.06592)

# Sparse Activation Functions
This repository contains the python package for **Sparse Activation Functions** which is used to generate the results of the paper **Sparsely Activated Networks** appeared in TNNLS.
See [Sparsely Activated Networks](https://github.com/pbizopoulos/sparsely-activated-networks).

# Example

```python
import torch
import sparse_activation_functions_pytorch as saf

minimum_extrema_distance = 10
extrema1d = saf.Extrema1D(minimum_extrema_distance)
input = torch.randn(1, 1, 100)
output = extrema2d(input)
```
