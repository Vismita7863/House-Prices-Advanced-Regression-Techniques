# final_pipeline.py

import pandas as pd
import numpy as np
import warnings
import optuna

# --- Import All Models and Tools ---
from sklearn.linear_model import Ridge
import catboost as cb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p
from scipy.optimize._optimize import BracketError

warnings.filterwarnings('ignore')

# ==============================================================================
# Part 1: Comprehensive Data Preparation (Inspired by Top Notebooks)
# ==============================================================================
print("\n[Part 1/5] Loading and preparing data from scratch...")

# --- 1.1 Load Raw Data ---
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    ames_df = pd.read_csv('AmesHousing.csv') # The external, larger dataset
    test_ids = test_df['Id']
except FileNotFoundError as e:
    print(f"Error: Raw data files not found. Ensure 'train.csv', 'test.csv', and 'AmesHousing.csv' are present.")
    exit()

# --- 1.2 Data Augmentation (Technique from the reference notebook) ---
print("  - Augmenting training data with external AmesHousing.csv...")
# Clean up column names to allow for merging
ames_df.columns = ames_df.columns.str.replace(' ', '').str.replace('/', '')
train_df.columns = train_df.columns.str.replace(' ', '').str.replace('/', '')
test_df.columns = test_df.columns.str.replace(' ', '').str.replace('/', '')
ames_df = ames_df.rename(columns={'PID': 'Id'})
ames_df = ames_df.drop('Order', axis=1)

# Combine competition train set with the parts of the Ames dataset not in the test set
full_train_df = pd.concat([train_df, ames_df.loc[~ames_df['Id'].isin(test_df['Id'])]]).reset_index(drop=True)
# Remove any exact duplicates that might have been created
full_train_df.drop_duplicates(inplace=True)
print(f"    Original train shape: {train_df.shape}. Augmented train shape: {full_train_df.shape}")

# --- 1.3 Outlier Removal ---
full_train_df = full_train_df.drop(full_train_df[(full_train_df['GrLivArea'] > 4000) & (full_train_df['SalePrice'] < 300000)].index).reset_index(drop=True)

# --- 1.4 Target Transformation and Data Combination ---
y = np.log1p(full_train_df["SalePrice"])
full_train_df = full_train_df.drop(['Id', 'SalePrice'], axis=1)
test_df = test_df.drop('Id', axis=1)
all_data = pd.concat((full_train_df, test_df)).reset_index(drop=True)

# --- 1.5 Impute Missing Values ---
print("  - Imputing missing values...")
for col in ('PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','MasVnrType'): all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','MasVnrArea'): all_data[col] = all_data[col].fillna(0)
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
for col in ('MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType','Utilities','Functional'): all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)

# --- 1.6 Feature Engineering ---
print("  - Engineering new features...")
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + 0.5 * all_data['HalfBath'] + all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']
all_data['TotalPorchSF'] = all_data['OpenPorchSF'] + all_data['3SsnPorch'] + all_data['EnclosedPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF']
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']
all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int); all_data['Has2ndFlr'] = (all_data['2ndFlrSF'] > 0).astype(int)
all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int); all_data['HasBsmt'] = (all_data['TotalBsmtSF'] > 0).astype(int)
all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)

# --- 1.7 Encoding and Transformation ---
print("  - Encoding and transforming features...")
ordinal_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5,'No': 1, 'Mn': 2, 'Av': 3, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6, 'RFn': 2, 'Fin': 3}
cols_ordinal = ['ExterQual', 'ExterCond', 'BsmtQual', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageFinish']
for col in cols_ordinal: all_data[col] = all_data[col].map(ordinal_map).fillna(0)
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str); all_data['YrSold'] = all_data['YrSold'].astype(str); all_data['MoSold'] = all_data['MoSold'].astype(str)
numerical_feats = all_data.select_dtypes(include=np.number).columns
skewed_feats = all_data[numerical_feats].apply(lambda x: skew(x.dropna())); skewed_feats = skewed_feats[abs(skewed_feats) > 0.5].index
for feat in skewed_feats:
    if len(all_data[feat].unique()) == 1: continue
    try: lam = boxcox_normmax(all_data[feat] + 1); all_data[feat] = boxcox1p(all_data[feat], lam)
    except (ValueError, BracketError): all_data[feat] = np.log1p(all_data[feat])

final_data = pd.get_dummies(all_data).reset_index(drop=True)

# --- 1.8 Post-Transformation Cleanup (This prevents all NaN/inf errors) ---
final_data.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = SimpleImputer(strategy='median')
final_data = imputer.fit_transform(final_data)
final_data = pd.DataFrame(final_data, columns=imputer.get_feature_names_out())

# --- 1.9 Final Split and Scaling ---
X = final_data.iloc[:len(y)]; X_test = final_data.iloc[len(y):]
scaler = StandardScaler(); X_scaled = scaler.fit_transform(X); X_test_scaled = scaler.transform(X_test)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
print("Data preparation complete.")

# ==============================================================================
# Part 2: Hyperparameter Tuning for Top Models
# ==============================================================================
print("\n[Part 2/5] Hyperparameter tuning for top models...")
def objective_catboost(trial):
    params = {'iterations': trial.suggest_int('iterations', 1000, 5000),'depth': trial.suggest_int('depth', 4, 8),'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True)}
    model = cb.CatBoostRegressor(**params, loss_function='RMSE', verbose=0, random_seed=42)
    score = np.mean(np.sqrt(-cross_val_score(model, X, y, cv=kfold, n_jobs=-1, scoring='neg_mean_squared_error')))
    return score
print("  - Tuning CatBoost..."); study_cat = optuna.create_study(direction='minimize'); study_cat.optimize(objective_catboost, n_trials=100)
print(f"    Best CatBoost CV Score: {study_cat.best_value:.5f}")

def objective_ridge(trial):
    alpha = trial.suggest_float('alpha', 10, 100, log=True)
    model = Ridge(alpha=alpha, random_state=42)
    score = np.mean(np.sqrt(-cross_val_score(model, X_scaled, y, cv=kfold, n_jobs=-1, scoring='neg_mean_squared_error')))
    return score
print("  - Tuning Ridge..."); study_ridge = optuna.create_study(direction='minimize'); study_ridge.optimize(objective_ridge, n_trials=50)
print(f"    Best Ridge CV Score: {study_ridge.best_value:.5f}")

# ==============================================================================
# Part 3: Train Final Models
# ==============================================================================
print("\n[Part 3/5] Training final models with the best parameters found...")
tuned_catboost = cb.CatBoostRegressor(**study_cat.best_params, loss_function='RMSE', verbose=0, random_seed=42)
tuned_ridge = Ridge(**study_ridge.best_params, random_state=42)
tuned_catboost.fit(X, y); tuned_ridge.fit(X_scaled, y)
print("Final models trained.")

# ==============================================================================
# Part 4: Generate All Final Submissions
# ==============================================================================
print("\n[Part 4/5] Generating all final submission files...")
preds_cat_log = tuned_catboost.predict(X_test); preds_ridge_log = tuned_ridge.predict(X_test_scaled)
final_preds_cat = np.expm1(preds_cat_log); pd.DataFrame({'Id': test_ids, 'SalePrice': final_preds_cat}).to_csv('submission_tuned_catboost.csv', index=False)
print("  ✅ Submission file 'submission_tuned_catboost.csv' created.")
final_preds_ridge = np.expm1(preds_ridge_log); pd.DataFrame({'Id': test_ids, 'SalePrice': final_preds_ridge}).to_csv('submission_tuned_ridge.csv', index=False)
print("  ✅ Submission file 'submission_tuned_ridge.csv' created.")
cat_score = study_cat.best_value; ridge_score = study_ridge.best_value; w_cat = 1 / cat_score; w_ridge = 1 / ridge_score; total_w = w_cat + w_ridge
final_preds_log_ensemble = (w_cat/total_w * preds_cat_log) + (w_ridge/total_w * preds_ridge_log)
final_predictions_ensemble = np.expm1(final_preds_log_ensemble)
pd.DataFrame({'Id': test_ids, 'SalePrice': final_predictions_ensemble}).to_csv('submission_ensemble_final.csv', index=False)
print("  ✅ Final ENSEMBLE submission file 'submission_ensemble_final.csv' created.")



