# German Credit Risk Prediction

A comprehensive machine learning project that predicts credit default risk using XGBoost, achieving 92.39% AUC with explainable AI.

## Project Overview

### Business Context
Credit risk assessment is critical for banking institutions to minimize loan defaults. This project develops an automated ML system to predict credit risk with high accuracy and transparency.

### Objectives
1. **Predict** credit default risk with 92%+ accuracy
2. **Explain** model decisions using SHAP analysis
3. **Process** features efficiently with missing data handling
4. **Compare** multiple ML algorithms for optimal performance

## Technical Implementation

### Data Pipeline
```
Raw Data → EDA → Feature Engineering → Model Training → SHAP Analysis → Model Saving
```

### Key Features
- **Missing Data Handling**: Smart strategies for 39.4% missing values
- **Feature Engineering**: Created 4 new predictive features
- **Multi-Algorithm Comparison**: Logistic Regression, Random Forest, XGBoost
- **Hyperparameter Optimization**: GridSearchCV with 5-fold cross-validation
- **Explainable AI**: SHAP TreeExplainer for model interpretability

## Results Achieved

### Model Performance
| Metric | Logistic Regression | Random Forest | **XGBoost (Best)** |
|--------|---------------------|---------------|---------------------|
| **ROC-AUC** | 91.7% | 89.64% | **92.39%** |
| **Accuracy** | 83.5% | 81.0% | **83.5%** |

### Dataset Characteristics
- **Size**: 1000 customers × 10 features → 29 features (after encoding)
- **Target Distribution**: 39.7% default rate (397 bad, 603 good)
- **Missing Values**: 39.4% checking_account, 18.3% saving_accounts
- **Train/Test Split**: 800/200 samples

### Feature Engineering Implemented
- One-hot encoding for categorical variables
- StandardScaler normalization for numerical features
- Created derived features: age_groups, amount_groups, duration_groups, amount_per_month
- Missing value imputation with 'unknown' category

### SHAP Analysis Results
**Top Risk Factors**:
1. **Credit Amount** (±2.0 impact)
2. **Age** (±0.8 impact)
3. **Job Category** (±0.7 impact)
4. **Monthly Payment Ratio** (±0.6 impact)

## Project Structure

```
german-credit-risk/
├── data/
│   └── german_credit_data_complete.csv     # Original dataset (1000 customers, 10 features)
├── notebooks/
│   └── 01_credit_risk_complete_analysis.ipynb  # Complete ML pipeline
├── models/                              # Saved models (created after running notebook)
│   ├── final_credit_risk_model.pkl         # Best XGBoost model
│   ├── feature_scaler.pkl                  # StandardScaler
│   ├── target_encoder.pkl                  # LabelEncoder
│   └── model_results.txt                   # Performance metrics
├── requirements.txt                        # Python dependencies
└── README.md                              # This documentation
```

**Notes for GitHub**:
- `models/` folder is created when running the notebook - contains trained models
- All visualizations are generated inline within the notebook (no separate files)
- Personal/sensitive folders (.venv, .vscode, results) are excluded via .gitignore

## Quick Start### Requirements
- Python 3.8+ (tested with Python 3.13)
- Git
- 4GB RAM minimum (8GB recommended)

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/AG-tech92/german-credit-risk.git
cd german-credit-risk

# 2. Create virtual environment (recommended)
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch Jupyter Notebook
jupyter notebook notebooks/01_credit_risk_complete_analysis.ipynb
```

### Alternative: Direct Notebook Installation
The notebook includes automatic dependency installation:
1. Open the notebook in Jupyter
2. Run the first few cells to install and verify dependencies
3. Follow the setup instructions in the notebook

### Alternative Installation (without requirements.txt)
```bash
pip install pandas>=2.3.0 scikit-learn>=1.7.0 xgboost>=3.0.0 shap>=0.48.0 matplotlib seaborn jupyter
```

### Validation
After installation, verify everything works:
```bash
# Check Python and key libraries
python -c "import pandas, sklearn, xgboost, shap, matplotlib; print('All dependencies installed successfully!')"

# Test Jupyter installation
jupyter --version

# Launch the analysis
jupyter notebook notebooks/01_credit_risk_complete_analysis.ipynb
```

## Technical Stack

- **Python 3.13** - Programming language
- **Pandas 2.3.1** - Data manipulation
- **Scikit-learn 1.7.1** - Machine learning algorithms
- **XGBoost 3.0.4** - Gradient boosting
- **SHAP 0.48.0** - Model explainability
- **Matplotlib/Seaborn** - Visualizations

## Methodology

### 1. Data Exploration
- Analyzed 1000 customers with 10 features
- Identified missing value patterns and distributions
- Computed correlation matrix and statistical summaries

### 2. Data Preprocessing
- Missing value handling with categorical imputation
- Feature engineering for age, amount, and duration groups
- One-hot encoding and standardization

### 3. Model Development
- Implemented 3 algorithms with baseline performance
- Hyperparameter optimization using GridSearchCV
- 5-fold cross-validation for robust evaluation

### 4. Model Evaluation
- ROC-AUC as primary metric (handles class imbalance)
- Multiple metrics: accuracy, precision, recall
- SHAP analysis for feature importance and interpretability

## Dataset Details

| Feature | Type | Description | Missing % |
|---------|------|-------------|-----------|
| Age | Numerical | Customer age (19-75) | 0% |
| Sex | Categorical | Male (69%), Female (31%) | 0% |
| Job | Numerical | Job category (0-3) | 0% |
| Housing | Categorical | Own/Rent/Free | 0% |
| Saving accounts | Categorical | Little/Moderate/Rich | **18.3%** |
| Checking account | Categorical | Little/Moderate/Rich | **39.4%** |
| Credit amount | Numerical | €250-€18,424 | 0% |
| Duration | Numerical | 4-72 months | 0% |
| Purpose | Categorical | Car/Electronics/etc. | 0% |
| Risk | Binary | Good (60.3%), Bad (39.7%) | 0% |

## Skills Demonstrated

- **Data Science**: End-to-end ML pipeline development
- **Feature Engineering**: Creating predictive features from raw data
- **Model Optimization**: Hyperparameter tuning and cross-validation
- **Explainable AI**: SHAP analysis for model interpretability

