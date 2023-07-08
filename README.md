![GitHub top language](https://img.shields.io/github/languages/top/siegert-lab/MorphOMICs.svg?style=for-the-badge)
[![GitHub](https://img.shields.io/github/license/siegert-lab/MorphOMICs.svg?style=for-the-badge)](https://github.com/siegert-lab/MorphOMICs/blob/master/license.txt)
[![GitHub contributors](https://img.shields.io/github/contributors/siegert-lab/MorphOMICs.svg?style=for-the-badge)](https://github.com/siegert-lab/MorphOMICs/graphs/contributors)
![GitHub repo size in bytes](https://img.shields.io/github/repo-size/siegert-lab/MorphOMICs.svg?style=for-the-badge)
[![GitHub issues](https://img.shields.io/github/issues/siegert-lab/MorphOMICs.svg?style=for-the-badge)](https://github.com/siegert-lab/MorphOMICs/issues)

# MorphOMICs


| ❗WARNING❗  |
|:------------------|
| These codes contain outdated scripts for running morphOMICs, but will still remain public for legacy reasons. The latest codes are found in https://github.com/siegert-lab/morphOMICs_v2 |                                    


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
git clone https://github.com/siegert-lab/MorphOMICs.git
cd morphomics
python3 setup.py install
```

# Usage
The easiest way to navigate through `MorphOMICs` is to run the `Morphomics_demo notebook`:
  - download `demo.zip` from https://seafile.ist.ac.at/f/eb13e707041749269ff9/?dl=1
  - unzip `demo.zip`
  - `cd demo`
  - `jupyter notebook`
  - Copy the url it generates, it looks something like this: `http://127.0.0.1:8888/?token=a4d016c37e162499e17b2993e69073fac0018bd9a779b762`
  - Open it in your browser
  - Then open `Morphomics_demo.ipynb`
