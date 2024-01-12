# FastVSDF
Code for FastVSDF: An Efficient Spatiotemporal Data Fusion Method for Seamless Data Cube

## Overview
FastVSDF is an efficient spatiotemporal data fusion method. 
It takes a fine/coarse image pair at T1 and coarse image at T2 to predict fine image at T2:
![image](https://github.com/ChenXuAxel/FastVSDF/assets/96739786/e0ec2602-403d-4efd-b056-428e7343c5e2)

## Get Started
### Prepare Environment [Python>=3.6]
1. Download source code from GitHub.
    ```sh
    git clone https://github.com/ChenXuAxel/FastVSDF
    
    cd DeepGuidedFilter && git checkout release
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
