import pandas as pd
import numpy as np
import warnings
import os

# --- Model Imports ---
from sklearn.linear_model import Lasso, Ridge
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# --- Stacking Library ---
# <-- CORRECTION: Using sklearn's native StackingRegressor instead of mlxtend
from sklearn.ensemble import StackingRegressor

# --- Utilities ---
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
print("--- Week 5 Script: Model Ensembling (Stacking & Blending) ---")

# Create submissions directory if it doesn't exist
os.makedirs('submissions', exist_ok=True)

# ==============================================================================
# Part 1: Load Data
# ==============================================================================
print("\n[Part 1/3] Loading data...")

try:
    X = pd.read_csv('train_featured.csv')
    X_test = pd.read_csv('test_featured.csv')
    
    # Load original training data to get the target variable 'SalePrice'
    train_df = pd.read_csv('train.csv') 
    
    # <-- CORRECTION: Load original test.csv to get the Test IDs
    test_df = pd.read_csv('test.csv') 
    
    y = train_df['SalePrice']
    
    # <-- CORRECTION: Get test IDs from the original test_df, not X_test
    test_ids = test_df['Id'] 
    
    # Log-transform the target variable
    y_log = np.log1p(y)

    # Drop 'Id' columns if they exist in the featured sets
    if 'Id' in X.columns:
        X = X.drop('Id', axis=1)
    if 'Id' in X_test.columns:
        X_test = X_test.drop('Id', axis=1)
        
    # Align columns - just in case
    # Get list of columns present in both, preserving order from X
    common_cols = [col for col in X.columns if col in X_test.columns]
    X = X[common_cols]
    X_test = X_test[common_cols]
    
    print(f"   Training features shape: {X.shape}")
    print(f"   Test features shape: {X_test.shape}")

except FileNotFoundError as e:
    # <-- CORRECTION: Added 'test.csv' to the error message
    print(f"Error: {e}. Make sure 'train_featured.csv', 'test_featured.csv', 'train.csv', and 'test.csv' are present.")
    exit()

# Scale data for linear models
# GBDTs are not sensitive to scaling, but Lasso/Ridge are.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


# ==============================================================================
# Part 2: (Core) Stacking Ensemble
# ==============================================================================
print("\n[Part 2/3] Building Stacking Ensemble...")

# --- 1. Define Base Models (Level 0) ---
lasso = Lasso(alpha=0.0004, random_state=42, max_iter=2000) 
lgbm = lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.05,
                         n_estimators=1000, max_bin=55, bagging_fraction=0.8,
                         bagging_freq=5, feature_fraction=0.23, feature_fraction_seed=9,
                         bagging_seed=9, min_data_in_leaf=6, min_sum_hessian_in_leaf=11,
                         random_state=42, n_jobs=-1)
xgboost = xgb.XGBRegressor(learning_rate=0.01, n_estimators=3000, max_depth=3,
                           min_child_weight=0, gamma=0, subsample=0.7,
                           colsample_bytree=0.7, objective='reg:squarederror',
                           nthread=-1, seed=42, reg_alpha=0.00006)
catboost = cb.CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=4,
                                l2_leaf_reg=1, random_seed=42, verbose=0,
                                early_stopping_rounds=100)

# --- 2. Define Meta-Model (Level 1) ---
meta_model_lasso = Lasso(alpha=0.0001, random_state=42, max_iter=2000)

# --- 3. Instantiate StackingRegressor ---
print("   Initializing StackingRegressor...")

# <-- CORRECTION: Define estimators as a list of (name, model) tuples for sklearn
estimators = [
    ('lasso', lasso),
    ('lgbm', lgbm),
    ('xgb', xgboost),
    ('cat', catboost)
]

# <-- CORRECTION: Use sklearn.ensemble.StackingRegressor
stack_reg = StackingRegressor(
    estimators=estimators,
    final_estimator=meta_model_lasso,  # Renamed from meta_regressor
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    passthrough=True,  # This is the equivalent of 'use_features_in_secondary=True'
    n_jobs=-1,
    verbose=1
)

# --- 4. Fit and Predict ---
print("   Fitting stacker (this may take several minutes)...")
# Pass unscaled data as a NumPy array for compatibility
stack_reg.fit(np.array(X), y_log)

print("   Predicting on test set...")
stack_preds_log = stack_reg.predict(np.array(X_test))

# --- 5. Inverse Transform and Submit ---
stack_preds = np.expm1(stack_preds_log)

submission_stack = pd.DataFrame({'Id': test_ids, 'SalePrice': stack_preds})
submission_stack.to_csv('submissions/submission_stacking.csv', index=False)
print("     Submission file 'submissions/submission_stacking.csv' created.")


# ==============================================================================
# Part 3: (Optional) Blending
# ==============================================================================
print("\n[Part 3/3] Performing simple blending of Week 4 submissions...")

try:
    sub_lgb = pd.read_csv('submission_LightGBM.csv')
    sub_cat = pd.read_csv('submission_CatBoost.csv')
    sub_lasso = pd.read_csv('submission_Lasso.csv')

    # --- Define Weights ---
    lgb_weight = 0.45
    cat_weight = 0.45
    lasso_weight = 0.10
    print(f"   Using weights: LGB={lgb_weight}, CatBoost={cat_weight}, Lasso={lasso_weight}")

    # --- Calculate Blend ---
    blended_price = (sub_lgb['SalePrice'] * lgb_weight) + \
                      (sub_cat['SalePrice'] * cat_weight) + \
                      (sub_lasso['SalePrice'] * lasso_weight)

    # --- Save Submission ---
    submission_blend = pd.DataFrame({'Id': test_ids, 'SalePrice': blended_price})
    submission_blend.to_csv('submissions/submission_blending.csv', index=False)
    print("     Submission file 'submissions/submission_blending.csv' created.")

except FileNotFoundError:
    print("   Skipping blending: Could not find all required submission files.")


# ==============================================================================
#                             END OF WEEK 5
# ==============================================================================

print("\n\n--- Week 5 Summary ---")
print("Deliverables Created:")
print("   - Stacking ensemble model fitted.")
print("   - Stacking submission file ('submissions/submission_stacking.csv')")
print("   - (Optional) Blended submission file ('submissions/submission_blending.csv')")
print("\nYou are now ready for Week 6.")