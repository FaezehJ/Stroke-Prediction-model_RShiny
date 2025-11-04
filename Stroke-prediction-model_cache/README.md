
# Stroke Prediction — R Shiny

An interactive **R Shiny** app to **train, evaluate, explain, and learn** stroke-risk models on tabular health data.  
Models: **Random Forest** and **Lasso Logistic (glmnet)** via **tidymodels**.

> ⚠️ Educational tool only. Not medical advice. Do not use for clinical decision-making.

-----------------------------------------------------------
## Features

- **Upload CSV** (local only). Auto name-cleaning, BMI coercion, robust target recoding (0/1 or yes/no).
- **Train/Test split** with optional **SMOTE** for class imbalance.
- **Models:** Random Forest (class weights & trees) or Lasso Logistic.
- **Metrics:** ROC AUC, PR AUC, accuracy / sensitivity / specificity / PPV / NPV @ 0.50 and at a user-selected threshold.
- **Curves:** ROC and Precision–Recall.
- **Explainability:**
  - Global: RF feature importance.
  - Local (Individual tab): “What changed the prediction?” leave-one-feature-out delta plot.
- **Individual predictions:** Enter age/BMI/glucose/etc.; get probability & class at the chosen threshold.
- **Export:** Download test-set predictions as CSV.

----------------------------------------------------------

### Requirements
- R ≥ 4.2 (tested on 4.5.x)
- RStudio recommended

### Install packages (reproducible way)
```r
install.packages("renv")
renv::restore()   # restores packages from renv.lock

If not using renv, install key packages:
install.packages(c(
  "shiny","shinythemes","DT","plotly",
  "tidymodels","themis","janitor","vip",
  "dplyr","ggplot2","readr","stringr"
))

click "Run App" in RStudio

-----------------------------------------------------------
                                         
Data
This repo does not include the dataset. See data/README.md for instructions to download the commonly used Kaggle Stroke Prediction Dataset (healthcare-dataset-stroke-data.csv) and where to place it.
The app ignores an id column if present.
Please review the data’s license/terms before use.

Data (not tracked in Git)

This project does not commit raw data. Put your local CSVs in the data/ folder and keep them out of version control (see .gitignore).

Source: Healthcare Dataset – Stroke Data (Kaggle; author: fedesoriano).
Kaggle page: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

If the link moves, search Kaggle for “Healthcare Dataset Stroke Data fedesoriano”.

Download & save locally

Download the CSV from the Kaggle page (requires a free Kaggle account).

Save the file to the data/ folder with this exact name:



data/healthcare-dataset-stroke-data.csv

The app expects columns like: gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status, stroke.


The app normalizes column names and coerces bmi to numeric; the target accepts 0/1 or no/yes.

How to use

Data: Upload CSV, confirm target (stroke). Check class balance and split sizes.
Model: Choose Random Forest (set trees, class weight) or Lasso Logistic; optionally enable SMOTE.
Train: Click Train.
Metrics: Review AUCs and class metrics at 0.50 and your threshold (left slider). Confusion matrix updates with the threshold.
Curves: Inspect ROC and PR.
Importance (RF): View top features.
Individual: Enter a single person’s values → Predict to see probability & class, plus a local explanation.
Predictions: Download test-set predictions as CSV.

Repository notes
.gitignore excludes large data and local outputs. See data/README.md for data instructions.
Consider renv::init() to lock dependencies for reproducibility.

License
MIT License for the code (see LICENSE). 
Data is licensed by its owner; see data/README.md for source and terms. No affiliation with MIT.

Acknowledgements
Built with tidymodels, themis, vip, shiny, DT, plotly, janitor.
Example schema based on the public “Stroke Prediction Dataset”.
