import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
import joblib
import warnings

from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso, RidgeCV
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
RANDOM_STATE = 42
FIGURES_DIR = 'figures'
MODELS_DIR = 'models'

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=== Starting Week 6 Analysis ===")

print("[1/6] Loading Data...")
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    print(f"  Train shape: {train_df.shape}, Test shape: {test_df.shape}")
except FileNotFoundError:
    print("  Error: 'train.csv' or 'test.csv' not found.")
    exit()

print("  Generating SalePrice distribution plot...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(train_df['SalePrice'], kde=True, color='blue')
plt.title('Original SalePrice Distribution')

plt.subplot(1, 2, 2)
sns.histplot(np.log1p(train_df['SalePrice']), kde=True, color='green')
plt.title('Log1p Transformed SalePrice')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/SalePrice_distribution.png')
plt.close()

print("[2/6] Preprocessing Data...")

train_df = train_df[~((train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000))]

train_ids = train_df['Id']
test_ids = test_df['Id']

y_train = np.log1p(train_df['SalePrice'])
train_df = train_df.drop(['SalePrice', 'Id'], axis=1)
test_df = test_df.drop(['Id'], axis=1)

ntrain = train_df.shape[0]
ntest = test_df.shape[0]
all_data = pd.concat([train_df, test_df]).reset_index(drop=True)

none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in none_cols:
    all_data[col] = all_data[col].fillna('None')

zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars',
             'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 
             'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
for col in zero_cols:
    all_data[col] = all_data[col].fillna(0)

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))

for col in all_data.columns:
    if all_data[col].isnull().sum() > 0:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + 0.5 * all_data['HalfBath'] + all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']
all_data['HasPool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['Has2ndFloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

all_data = pd.get_dummies(all_data)

X_train = all_data[:ntrain]
X_test = all_data[ntrain:]

print(f"  Final Feature Count: {X_train.shape[1]}")
print("[3/6] Defining Models & Stacking Architecture...")

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

ridge = Pipeline([('scaler', RobustScaler()), ('model', Ridge(alpha=10, random_state=RANDOM_STATE))])
lasso = Pipeline([('scaler', RobustScaler()), ('model', Lasso(alpha=0.0005, random_state=RANDOM_STATE))])
lgbm = LGBMRegressor(objective='regression', n_estimators=1000, learning_rate=0.05, verbose=-1, random_state=RANDOM_STATE)
cat = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, verbose=0, allow_writing_files=False, random_state=RANDOM_STATE)
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1, random_state=RANDOM_STATE)

stack_gen = StackingRegressor(
    estimators=[
        ('cat', cat),
        ('ridge', ridge),
        ('lgbm', lgbm),
        ('lasso', lasso)
    ],
    final_estimator=RidgeCV(),
    cv=kf,
    n_jobs=-1
)

print("[4/6] Evaluating Baseline Models for Report Figures...")

models = {'Ridge': ridge, 'Lasso': lasso, 'LGBM': lgbm, 'CatBoost': cat, 'XGBoost': xgb}
results = []
names = []
std_devs = []

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=kf, n_jobs=-1)
    rmse_scores = -cv_scores
    results.append(rmse_scores.mean())
    std_devs.append(rmse_scores.std())
    names.append(name)
    print(f"  {name}: Mean RMSE: {rmse_scores.mean():.4f} (Std: {rmse_scores.std():.4f})")

plt.figure(figsize=(10, 6))
sns.barplot(x=names, y=results, palette='viridis')
plt.title('Model Performance Comparison (CV RMSE)')
plt.ylabel('Root Mean Squared Error (Log Scale)')
plt.savefig(f'{FIGURES_DIR}/model_performance_comparison.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(x=names, y=std_devs, palette='magma')
plt.title('Model Stability Comparison (CV RMSE Std Dev)')
plt.ylabel('Standard Deviation of RMSE')
plt.savefig(f'{FIGURES_DIR}/model_stability_comparison.png')
plt.close()

print("[5/6] Training Final Stacking Model...")

stack_gen.fit(X_train, y_train)
print("  Stacking Model Trained.")

joblib.dump(stack_gen, f'{MODELS_DIR}/week6_model.pkl')

print("  Generating Predictions...")
cat.fit(X_train, y_train) 
preds_stack = stack_gen.predict(X_test)
preds_cat = cat.predict(X_test)
final_preds = np.expm1(0.7 * preds_stack + 0.3 * preds_cat)

sub = pd.DataFrame()
sub['Id'] = test_ids
sub['SalePrice'] = final_preds
sub.to_csv('submission_stacking.csv', index=False)
print("  Saved 'submission_stacking.csv'")

try:
    print("\n  Stacking Meta-Model Weights:")
    print(f"  Intercept: {stack_gen.final_estimator_.intercept_:.4f}")
    print(f"  Weights: {stack_gen.final_estimator_.coef_}")
except:
    print("  Could not retrieve weights directly (depends on meta-learner structure).")

print("[6/6] Running Interpretability Analysis...")

explainer = shap.TreeExplainer(cat)
shap_values = explainer.shap_values(X_train)

plt.figure()
shap.summary_plot(shap_values, X_train, plot_type="bar", show=False, max_display=20)
plt.title("SHAP Global Feature Importance")
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/shap_summary_bar.png')
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_train, show=False, max_display=20)
plt.title("SHAP Summary Beeswarm")
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/shap_summary_beeswarm.png')
plt.close()

vals = np.abs(shap_values).mean(0)
feature_importance = pd.DataFrame(list(zip(X_train.columns, vals)), columns=['col_name','feature_importance_vals'])
feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
top_feature = feature_importance.iloc[0]['col_name'] 
print(f"  Top SHAP feature: {top_feature}")

target_feat = 'OverallQual' if 'OverallQual' in X_train.columns else top_feature

plt.figure()
shap.dependence_plot(target_feat, shap_values, X_train, show=False)
plt.title(f"SHAP Dependence: {target_feat}")
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/shap_dependence_{target_feat}.png')
plt.close()

print("  Generating LIME explanation...")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    class_names=['SalePrice'],
    mode='regression',
    random_state=RANDOM_STATE
)

idx = 0
exp = lime_explainer.explain_instance(
    data_row=X_test.iloc[idx].values, 
    predict_fn=cat.predict
)

fig = exp.as_pyplot_figure()
plt.title(f"LIME Explanation for Test Instance {idx}")
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/lime_explanation.png')
plt.close()

print("  Calculating Permutation Importance (on Ridge model for contrast)...")

ridge.fit(X_train, y_train) 

perm_importance = permutation_importance(ridge, X_train, y_train, n_repeats=10, random_state=RANDOM_STATE)
sorted_idx = perm_importance.importances_mean.argsort()[-20:]

plt.figure(figsize=(10, 8))
plt.barh(X_train.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title("Permutation Importance (Ridge Model)")
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/permutation_importance.png')
plt.close()

feature_importance.head(20).to_csv('interpretability_top_shap_features.csv', index=False)

print("\n=== Week 6 Analysis Complete ===")
print(f"Figures saved in: {os.path.abspath(FIGURES_DIR)}")
print(f"Model saved in: {os.path.abspath(MODELS_DIR)}")
print("Submission ready.")