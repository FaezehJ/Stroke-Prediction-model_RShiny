# Stroke-Prediction-model_RShiny
Interactive R Shiny app to train, evaluate, and explain stroke-risk models (Random Forest &amp; Lasso Logistic) on tabular health data. Supports class imbalance handling (SMOTE), threshold tuning, ROC/PR curves, feature importance, single-patient predictions, and CSV export—powered by tidymodels.



# Stroke Prediction — R Shiny
An interactive **R Shiny** app to **train, evaluate, and explain** stroke-risk models on tabular health data.  
Models: **Random Forest** and **Lasso Logistic (glmnet)** via **tidymodels**.
> ⚠️ Educational tool only. Not medical advice. Do not use for clinical decision-making.

                                        -----------------------------------------------------------------
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

                                         -----------------------------------------------------------------
### Requirements
- R ≥ 4.2 (tested on 4.5.x)
- RStudio recommended

### Install packages (reproducible way)

install.packages("renv")
renv::restore()   # restores packages from renv.lock

If not using renv, install key packages:
install.packages(c(
  "shiny","shinythemes","DT","plotly",
  "tidymodels","themis","janitor","vip",
  "dplyr","ggplot2","readr","stringr"
))

click "Run App" in RStudio

-----------------------------------------------------------------
Data:
This repo does not include the dataset. See data/README.md for instructions to download the commonly used Kaggle Stroke Prediction Dataset (healthcare-dataset-stroke-data.csv) and where to place it.
The app ignores an id column if present.
Please review the data’s license/terms before use.


This project uses the **Stroke Prediction Dataset** (file: `healthcare-dataset-stroke-data.csv`) commonly distributed on Kaggle.

- Source: Kaggle — “Stroke Prediction Dataset” by fedesoriano  
  (Search for: kaggle fedesoriano stroke prediction dataset)
- Contents: 5,110 rows × 12 columns (id, gender, age, hypertension, heart_disease,
  ever_married, work_type, Residence_type/residence_type, avg_glucose_level, bmi,
  smoking_status, stroke).

## How to obtain
1) Visit the Kaggle dataset page and download `healthcare-dataset-stroke-data.csv`.
2) Place it in this folder as:  
   `data/healthcare-dataset-stroke-data.csv`

## License
The dataset is licensed by the *data owner* (see Kaggle page for the definitive license and terms).  
**Do not redistribute the CSV in this repository.** Cite the source when using the data.

### Suggested citation
> fedesoriano. *Stroke Prediction Dataset.* Kaggle. (Accessed YYYY-MM-DD).


Expected columns:
gender, age, hypertension (0/1), heart_disease (0/1), ever_married, work_type,
residence_type (raw may be Residence_type), avg_glucose_level, bmi, smoking_status, stroke (target).

Column	Type / values	Notes
gender	factor: Male, Female	
age	numeric	years
hypertension	0/1	
heart_disease	0/1	
ever_married	Yes/No	
work_type	e.g., Private,Self-employed,Govt_job,…	
residence_type	Urban/Rural	
avg_glucose_level	numeric	mg/dL
bmi	numeric (coerced if character)	
smoking_status	never smoked,formerly smoked,smokes,Unknown	
stroke	target, binary (0/1 or no/yes)	required

The app normalizes column names and coerces bmi to numeric; the target accepts 0/1 or no/yes.

How to use:
Data: Upload CSV, confirm target (stroke). Check class balance and split sizes.
Model: Choose Random Forest (set trees, class weight) or Lasso Logistic; optionally enable SMOTE.
Train: Click Train.
Metrics: Review AUCs and class metrics at 0.50 and your threshold (left slider). Confusion matrix updates with the threshold.
Curves: Inspect ROC and PR.
Importance (RF): View top features.
Individual: Enter a single person’s values → Predict to see probability & class, plus a local explanation.
Predictions: Download test-set predictions as CSV.

Repository notes:
.gitignore excludes large data and local outputs. See data/README.md for data instructions.
Consider renv::init() to lock dependencies for reproducibility.

License:
MIT License for the code (see LICENSE). 
Data is licensed by its owner; see data/README.md for source and terms. No affiliation with MIT.

Acknowledgements:
Built with tidymodels, themis, vip, shiny, DT, plotly, janitor.
Example schema based on the public “Stroke Prediction Dataset”.
