# New features

This document describes the new features in this Caffe branch.

## New layers

### Weighted Softmax Loss Layer

| Class name               | Prototxt name         |
|--------------------------|-----------------------|
| WeightedSoftmaxLossLayer | `WeightedSoftmaxLoss` |

Similar to `SOFTMAX_LOSS`, except it takes a third bottom input specifying the
importance of each sample, e.g.:

```
layers {
  name: "loss"
  type: WEIGHTED_SOFTMAX_LOSS
  bottom: "fc"
  bottom: "label"
  bottom: "sample_weight"
}
```

The shape of `sample_weight` should be `(N, 1, 1, 1)` or simply `(N,)`, where
`N` is the number of samples.

Note that the HDF5 loader, unlike in earlier releases of Caffe, can now load
any number of inputs with any key. That way, you can add `sample_weight` (or
whatever you wish to name it) to your data file:

```
layers {
  name: "main"
  type: HDF5_DATA
  top: "data"
  top: "label"
  top: "sample_weight"
  hdf5_data_param {
    source: "/path/to/data.txt"  # File should contain an absolute path to h5 file
    batch_size: 100
  }
}
```
This assumes that the HDF5 file has an entry at `/sample_weight`. You can also
load it separately from its own HDF5 file. I have not tested it it with lmdb,
but I think it will work analogously.

### Weighted Sigmoid Cross-entropy Loss Layer

| Class name                           | Prototxt name                 |
|--------------------------------------|-------------------------------|
| WeightedSigmoidCrossEntropyLossLayer | `WeightedSigmoidCrossEntropy` |

Similar to `SIGMOID_CROSS_ENTROPY_LOSS`, except it takes a third bottom input
specifying the importance of each sample, e.g.:

```
layers {
  name: "loss"
  type: WEIGHTED_SIGMOID_CROSS_ENTROPY_LOSS
  bottom: "fc"
  bottom: "label"
  bottom: "sample_weight"
}
```

The shape of `sample_weight` should be the same as `label`, which normally
means `(N, C, 1, 1)`, where `N` is the number of samples and `C` the number of
classes.

### Hypercolumn Layer

| Class name                | Prototxt name          |
|---------------------------|------------------------|
| HypercolumnExtractorLayer | `HypercolumnExtractor` |

Example:

```
layer {
  name: "hypercolumn"
  type: "HypercolumnExtractor"
  bottom: "centroids"
  bottom: "conv1"
  bottom: "conv2"
  bottom: "conv3"
  top: "columns"

  hypercolumn_extractor_param {
    scale: 2
    scale: 4
    scale: 8

    offset_height: 3
    offset_height: 4
    offset_height: 4

    offset_width: 3
    offset_width: 4
    offset_width: 4
  }
}
```
The first bottom input provides the hypercolumn locations and should be floats
of shape `(N, P, 2)`, where `P` is the number of hypercolumns. The last axis
provides the location (first vertical and then horizontal) in an arbitrarily
chosen reference coordinate system (the original image size is of course
recommended). The translation between a point, `p`, in the reference coordinate
system and in the local layer `local_p` is defined as `p = offset + local_p *
scale`. The offset and the scale is defined in the `hypercolumn_extract_param`
as in the example above.

All following bottom inputs are the layers from which to construct the
hypercolumn. You can use any number of layers. Make sure to specify the
coordinate translation parameters for each layer.

A hypercolumn is unlikely to land exactly on the grid of a layer's filter, so
bilinear interpolation is used to combined the closest four.

The top output will be of shape `(N * P, C)` where `C` is the sum of the
channels for all input layers. For instance, if `conv1` through `conv3` have
16, 32 and 64 filters respectively, then `C` will be 112.
