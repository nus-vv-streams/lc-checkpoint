# LC-Checkpoint for NeRF-based Volumetric Video Compression

This repository contains the implementations of the adapted LC-Checkpoint [1] for our work "Volumetric Video Compression Through Neural-based Representation" [[Paper](https://dl.acm.org/doi/10.1145/3652212.3652220)]. The code is written in Python.

## Overview

We propose an end-to-end pipeline for volumetric video compression using neural-based representation. In this pipeline, we represent 3D dynamic content as a sequence of NeRFs, converting the explicit representation to neural representation. 

<p align="center">
  <a href="">
    <img src="/fig/representation.png" alt="teaser" width="50%">
  </a>
</p>

Building on the insight of significant similarity between successive NeRFs, we propose to benefit from this temporal coherence: we encode the differences between consecutive NeRFs, achieving substantial bitrate reduction without noticeable quality loss.

<p align="center">
  <a href="">
    <img src="/fig/pipeline.png" alt="teaser" width="50%">
  </a>
</p>

We adapted an efficient and scalable model compression scheme, LC-Checkpoint, in our proposed compression pipeline.


## Instructions

The implementation is tested with Python 3.9.16 and PyTorch 2.1.1.

### Installation

To install the required packages, you can run the following command:

```bash
pip install -r requirements.txt
```

### Usage

To use the code, you can clone the repository and import the main script:

```python
import src.main as lc
```

To compress a sequence of NeRFs, you can run the following command:

```python
lc.compress_set(filename=model_dir, models=enc_model_list, saveloc=COMPRESSED_SAVELOC, num_bits=num_bits)
```

where `model_dir` is the directory of the NeRF models, `enc_model_list` is the list of the encoder models, `COMPRESSED_SAVELOC` is the directory to save the compressed sequence, and `num_bits` is the number of bits for bucket indexing for exponentbased quantization. We trace the performance of the compressed model at different bitrates by changing the `num_bits`.


To decompress the compressed sequence, you can run the following command:

```python
lc.load_compressed_set(COMPRESSED_SAVELOC, dec_model_list, DECOMPRESSED_SAVELOC, BASE_DICT)
```

where `COMPRESSED_SAVELOC` is the directory of the compressed sequence, `dec_model_list` is the list of the decoder models, `DECOMPRESSED_SAVELOC` is the directory to save the decompressed sequence, and `BASE_DICT` is the base dictionary.

### Example

We provide an example ipynb file `example.ipynb` to compress and decompress an example sequence of NeRFs. The example NeRFs were trained by following the proposed pipeline on a dynamic point cloud `RedAndBlack` from [8iVFB Dataset](http://plenodb.jpeg.org/pc/8ilabs/). 

## Reference

[1] Chen, Y., Liu, Z., Ren, B., & Jin, X. (2020). On efficient constructions of checkpoints. arXiv preprint arXiv:2009.13003.

## Citation

If you find this code useful for your research, please consider citing the following paper:

```latex
@inproceedings{shi2024volumetric,
  title={Volumetric Video Compression Through Neural-based Representation},
  author={Shi, Yuang and Zhao, Ruoyu and Gasparini, Simone and Morin, G{\'e}raldine and Ooi, Wei Tsang},
  booktitle={Proceedings of the 16th International Workshop on Immersive Mixed and Virtual Environment Systems},
  pages={85--91},
  year={2024}
}
```