Project 1: Use of AI Fairness 360 (AIF360) Package on Financial Credit Dataset
==============================================================================

NJIT Data Science Program
Course
CS634 Data Mining - Summer 2020
Instructor
Pantelis Monogioudis, Ph.D Professor of Practice, NJIT & Adjunct NYU
Teaching Assistant
Michael Lam (PhD student)

Students:
Fernando Rios
Hassan Ouanir
Ian Kavuma


Description
The AI Fairness 360 toolkit is an extensible open-source library containg techniques developed by the research community to help detect and mitigate bias in machine learning models throughout the AI application lifecycle. The AI Fairness 360 Python package includes
a comprehensive set of metrics for datasets and models to test for biases,
explanations for these metrics, and algorithms to mitigate bias in datasets and models. 

We have developed this project from scratch, In the Question 1, we did an overview of the metrics and algorithms included in the AIF360 package. In question 2, we used the metric “mean difference” outcomes between unprivileged and privileged groups to determine the fairness. We then applied Reweighing algorithm for bias mitigation. For question 3, we used Optimized Preprocessing Algorithm for bias mitigation. And in question 4, we compared both results using Reweighting and Optimized Preprocessing algorithm. Finally, we incorporated an additional section called "additional work" to show other features that are necessary to understand the project better. These are, Training unbiased models using Logistic Regression, Calculating Disparate Impact metric on train dataset without AIF360 package. We analyzed the Original Dataset using graphics such as Histogram, Pair Plot and Correlation. Additionally, we analyzed the group age, the privileged and unprivileged attribute with favorable and unfavorable outcomes.


Setup
This notebook is prepared to run easily, it includes in its lines:
AIF360 package installer
	! pip install aif360

Setting up all necessary libraries
	import sys
	import numpy as np
	np.random.seed(0)
	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns
	from tqdm import tqdm

	from aif360.datasets import BinaryLabelDataset
	from aif360.datasets import GermanDataset
	from aif360.metrics import BinaryLabelDatasetMetric
	from aif360.metrics import ClassificationMetric
	from aif360.metrics.utils import compute_boolean_conditioning_vector
	from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
	from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_german
	from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_german
	from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
	from aif360.algorithms.preprocessing.reweighing import Reweighing
	from aif360.algorithms.preprocessing import DisparateImpactRemover

	from sklearn.linear_model import LogisticRegression
	from sklearn.preprocessing import StandardScaler
	from sklearn.metrics import accuracy_score

	from IPython.display import Markdown, display
	
	from aif360.explainers import MetricTextExplainer

  Copy of the original dataset german.data to google colab folder.  
	!wget -O ../usr/local/lib/python3.6/dist-packages/aif360/data/raw/german/german.data https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data

Contributions are welcome!
