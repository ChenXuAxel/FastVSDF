# FastVSDF
Code for FastVSDF: An Efficient Spatiotemporal Data Fusion Method for Seamless Data Cube

## Overview
**FastVSDF is an efficient spatiotemporal data fusion method.**

The purpose of proposing FastVSDF is to achieve data fusion at the lowest cost. Therefore, FastVSDF requires only minimal input data, while the computational cost needed is low.

It takes a fine/coarse image pair at T1 and coarse image at T2 to predict fine image at T2:
![image](https://github.com/ChenXuAxel/FastVSDF/assets/96739786/e0ec2602-403d-4efd-b056-428e7343c5e2)

## Get Started
### Prepare Environment [Python>=3.6]
1. Download source code from GitHub.
    ```sh
    git clone https://github.com/ChenXuAxel/FastVSDF
    
    cd FastVSDF && git checkout release
    ```
2. Install dependencies.
    ```sh
    conda install gdal guided_filter_pytorch scikit-learn scikit-image pytorch 
    ```
### Predict
   ```sh
   from FastVSDF import FastVSDF
   FastVSDF(F1_path, C1_path, C2_path, fastvsdf_path)
   ```

## Cite
If you find FastVSDF is helpful, please cite the following work:

1. FastVSDF [[Paper]](https://ieeexplore.ieee.org/document/10399795) [[Code]](https://github.com/ChenXuAxel/FastVSDF)
```
@ARTICLE{10399795,
  author={Xu, Chen and Du, Xiaoping and Fan, Xiangtao and Jian, Hongdeng and Yan, Zhenzhen and Zhu, Junjie and Wang, Robert},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={FastVSDF: An Efficient Spatiotemporal Data Fusion Method for Seamless Data Cube}, 
  year={2024},
  doi={10.1109/TGRS.2024.3353758}}
```

2. VSDF [[Paper]](https://www.sciencedirect.com/science/article/pii/S0034425722004151) [[Code]](https://github.com/ChenXuAxel/VSDF)
```
@article{XU2022113309,
title = {VSDF: A variation-based spatiotemporal data fusion method},
journal = {Remote Sensing of Environment},
volume = {283},
pages = {113309},
year = {2022},
issn = {0034-4257},
doi = {https://doi.org/10.1016/j.rse.2022.113309},
}
```

## Contact
If you have any question, please contact Chen Xu(xuchen@aircas.ac.cn) or submit a issue.
