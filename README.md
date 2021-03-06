Project 1: Use of AI Fairness 360 (AIF360) Package on Financial Credit Dataset
==============================================================================

NJIT Data Science Program Course
CS634 Data Mining - Summer 2020 <br>
Instructor Pantelis Monogioudis, Ph.D Professor of Practice, NJIT & Adjunct NYU<br>
Teaching Assistant Michael Lam (PhD student)


Students:<br> Fernando Rios <br> Hassan Ouanir <br> Ian Kavuma

Description

The AI Fairness 360 toolkit is an extensible open-source library containg techniques developed by the research communityto help detect and mitigate bias in machine learning models throughout the AI application lifecycle. The AI Fairness 360 Python package includes a comprehensive set of metrics for datasets and models to test for biases,
explanations for these metrics, and algorithms to mitigate bias in datasets and models. 

This project consists of four questions related to bias and mitigation on Financial Credit Dataset (GermanData). 
<br>In the question 1, we did an overview of the metrics and algorithms of bias mitigation, included in the AIF360 package. 
<br>In question 2, we used the metric "mean difference" outcome between unprivileged and privileged groups to determine the fairness. We then  applied Reweighing algorithm for bias mitigation.
<br>For question 3, we used Optimized Preprocessing Algorithm for bias mitigation with the same metric.
<br>In question 4, we compared both results using Reweighting and Optimized Preprocessing algorithms. The metrics and bias mitigation algorithms were used in train and test datasets.
Finally, we incorporated an additional section called "additional work" to show others features that are necessary to understand the project better.<br> These are, Training unbiased models using Logistic Regression, Calculating Disparate Impact metric on train dataset without AIF360 package. We analyzed the Original Dataset using graphics such as Histogram, Pair Plot and Correlation.<br> Additionally, we analyzed the group age, the privileged and unprivileged attribute with favorable and unfavorable outcomes.

### Setup 
This notebook is prepared to run easily, just select the option "run all" in google colab. 
In its lines this notebook install the AIF360 package, setting all libraries, copy the dataset file german.data in google colab, load and split the dataset.
### AIF360 package installer
```bash
	! pip install aif360
``` 
### Setting up all necessary libraries

```python 

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
```
Copy of the original dataset german.data to google colab folder.

``` python  
	!wget -O ../usr/local/lib/python3.6/dist-packages/aif360/data/raw/german/german.data https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
```
Loading  and split dataset 
``` python 
dataset_orig = GermanDataset(protected_attribute_names=['age'],privileged_classes=[lambda x: x >= 25],features_to_drop=['personal_status', 'sex']  dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]
```
After this, you can go through the all lines to read the description of each method applied to detect bias and determine bias mitigation.

### Contributing 
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
