# MorphOMICs

| :exclamation point: ANNOUNCEMENT                                                                   |
|:---------------------------------------------------------------------------------------------------|
| Due to a cyber attack on our Institute, the Git server (https://git.ist.ac.at/rcubero/morphomics/) |
| which hosted `MorphOMICs` was taken down from public access. While the Institute's infrastructures |
| are (slowly) being brought back to functionality, you can use this repository to download and try  |
| out the codes. In the meanwhile, we do ask for your patience.                                      |

Due to a cyber attack on our Institute, the Git server (https://git.ist.ac.at/rcubero/morphomics/) which hosted `MorphOMICs` was taken down from public access. While the Institute's infrastructures are (slowly) being brought back to functionality, you can use this repository to download and try out the codes. In the meanwhile, we do ask for your patience.

`MorphOMICs` is a Python package containing tools for analyzing microglia morphology using a topological data analysis approach. Note that this algorithm is designed not only for microglia applications but also for any dynamic branching structures across natural sciences.

- [Overview](#overview)
- [Required Dependencies](#required-dependencies)
- [Installation Guide](#installation-guide)
- [Usage](#usage)

# Overview
`MorphOMICs` is a novel approach which combines the Topological Morphology Descriptor (TMD) with bootstrapping approach, dimensionality reduction strategies to visualize microglial morphological signatures and their relationships across different biological conditions.


# Required Dependencies
Python : 3.7+

numpy : 1.8.1+, scipy : 0.13.3+, pickle : 4.0+, enum34 : 1.0.4+, scikit-learn : 0.19.1+, matplotlib : 3.2.0+

Additional dependencies:
anndata : 0.7+, umap-learn : 0.3.10+, palantir : 1.0.0+, fa2

# Installation Guide
```
git clone https://git.ist.ac.at/rcubero/morphomics
cd morphomics
python3 setup.py install
```

# Usage
The easiest way to navigate through `MorphOMICs` is to run the demo notebook:
  - `cd demo`
  - `jupyter notebook`
  - Copy the url it generates, it looks something like this: `http://127.0.0.1:8888/?token=a4d016c37e162499e17b2993e69073fac0018bd9a779b762`
  - Open it in your browser
  - Then open `Morphomics_demo.ipynb`
