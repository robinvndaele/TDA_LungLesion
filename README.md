************ FOR REVIEWING PURPOSES ONLY. PLEASE DO NOT DISTRIBUTE. ************

# Topological data analysis for tumor histology prediction

This folder contains the source code for Nature Machine Intelligence submission:

*Topological data analysis of CT scan images for lung tumor histology prediction*

# Setup

- Enviroments: Python (jupyter notebook)
- Requirements (tell my why):
	- numpy (handling arrays)
	- pandas (handling data frames)
	- scipy (obtaining summary statistics of persistence diagrams and p-values)
	- matplotlib (plotting)
	- seaborn (plotting)
	- dionysus (persistent homology of images)
	- ripser (persistent homology of point clouds)
	- persim (plotting persistent diagrams)
	- SimpleITK (handling dicom images)
	- imageio (handling images)
	- scikit-image (converting tumor segmentations to meshes)
	- plotly (3D plotting)
	- pymrmr (minimum redundancy maximum relevance feature selection)
	- scikit-learn (most machine learning models and to conduct our experiments)
	- xgboost (XGBoost models)

# Data

- One example tumor + segmentation is available from the "Scan" folder
- All persistence diagrams (SF/PA + LIDC) available from the "Diagram" folder
- All radiomic features and metadata are available from the "Features" folder
- Original scans for the LIDC data are available from https://pylidc.github.io/

# Run

In Python Jupyter notebook: Cell --> Run Cells (ctrl + enter).
Change the working directory in the first code block to your main repository path first.

- Scripts/TDAtumor.ipynb: Illustrates how we obtain all topological features from an example tumor (should only take some minutes)
- Experiments/SF-PA.ipynb: Conduct all experiments on SF/PA cohort (WARNING: can take a couple of hours)
- Experiments/LIDC.ipynb: Conduct all experiments on LIDC data (WARNING: can take a couple of hours)

# Output

The output contains all output by code block in html format, with files named accordingly.

# Contact

Robin.Vandaele@UGent.be
