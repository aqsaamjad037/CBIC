# CBIC
CBIC: Cross-cohort Blood IBD Classifier. Leave-one-dataset-out classification of IBD (IBD vs Control; CD vs UC) from whole-blood transcriptomes of five public cohorts.
## Data
Five whole-blood cohorts from GEO: GSE186507, GSE112057, GSE119600, GSE169568, GSE126124.
Colon cohorts used for the Level 2 benchmark: GSE75214, GSE179285.

## Requirements
Python 3.10.19, scikit-learn 1.7.1, pandas 2.3.3, NumPy 2.2.6 (see requirements.txt).

## Reproducing the figures
1. Put the data files in one folder (set IBD_DIR to that folder).
2. Run make_figures.py, make_figures_extra.py and make_circular.py.

## Citation
If you use this code, please cite the manuscript and the Zenodo archive (DOI: to be added).

## License
MIT
