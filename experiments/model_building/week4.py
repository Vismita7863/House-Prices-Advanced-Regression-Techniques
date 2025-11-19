import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Embedding, concatenate, Flatten
from keras.callbacks import EarlyStopping, LearningRateScheduler

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
print("--- Week 4 Master Script: Comprehensive Model Evaluation ---")

os.makedirs('plots', exist_ok=True)
os.makedirs('submissions', exist_ok=True)

print("\n[Part 1/6] Loading data and setting up...")
try:
    X_featured = pd.read_csv('train_featured.csv')
    X_test_featured = pd.read_csv('test_featured.csv')
    train_df_orig = pd.read_csv('train.csv')
    test_df_orig = pd.read_csv('test.csv')
    y = np.log1p(train_df_orig['SalePrice'])
    test_ids = test_df_orig['Id']
except FileNotFoundError as e:
    print(f"Error: Could not find required data files. Please run the Week 3 script first.")
    exit()

baseline_ridge_score = 0.1319
baseline_ridge_std = 0.0197
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model_scores = {}

print("\n[Part 2/6] Evaluating traditional ML models...")

scaler_trad = StandardScaler()
X_scaled_trad = scaler_trad.fit_transform(X_featured)
X_test_scaled_trad = scaler_trad.transform(X_test_featured)

traditional_models = {
    'Lasso': {'model': Lasso(alpha=0.0005, random_state=42), 'data': 'scaled', 'type': 'Core'},
    'RandomForest': {'model': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), 'data': 'unscaled', 'type': 'Core'},
    'LightGBM': {'model': lgb.LGBMRegressor(objective='regression', random_state=42), 'data': 'unscaled', 'type': 'Core'},
    'XGBoost': {'model': xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1), 'data': 'unscaled', 'type': 'Core'},
    'CatBoost': {'model': cb.CatBoostRegressor(loss_function='RMSE', verbose=0, random_seed=42), 'data': 'unscaled', 'type': 'Core'},
    'SVR': {'model': SVR(C=0.1, kernel='rbf'), 'data': 'scaled', 'type': 'Optional'} 
}   

for name, info in traditional_models.items():
    data_to_use = X_scaled_trad if info['data'] == 'scaled' else X_featured
    print(f"  - Cross-validating {name}...")
    cv_scores = np.sqrt(-cross_val_score(info['model'], data_to_use, y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1))
    model_scores[name] = {'mean': cv_scores.mean(), 'std': cv_scores.std()}
    print(f"    {name} CV RMSLE: {model_scores[name]['mean']:.4f} (+/- {model_scores[name]['std']:.4f})")

print("\n[Part 3/6] Evaluating Neural Network with Entity Embeddings...")

def gelu(x): return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
def cyclical_lr(epoch):
    base_lr, max_lr, step_size = 1e-4, 2e-3, 30
    cycle = np.floor(1 + epoch / (2 * step_size)); x = np.abs(epoch / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))

all_data_emb = pd.concat((train_df_orig.drop('SalePrice', axis=1), test_df_orig), ignore_index=True)
for col in ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','MasVnrType']: all_data_emb[col]=all_data_emb[col].fillna('None')
for col in ['GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','MasVnrArea']: all_data_emb[col]=all_data_emb[col].fillna(0)
all_data_emb['LotFrontage']=all_data_emb.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median()))
for col in ('MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType','Utilities','Functional'): all_data_emb[col]=all_data_emb[col].fillna(all_data_emb[col].mode()[0])
all_data_emb['TotalSF']=all_data_emb['TotalBsmtSF']+all_data_emb['1stFlrSF']+all_data_emb['2ndFlrSF']
all_data_emb['HouseAge']=all_data_emb['YrSold']-all_data_emb['YearBuilt']
all_data_emb['RemodAge']=all_data_emb['YrSold']-all_data_emb['YearRemodAdd']
cat_cols_emb = [c for c in all_data_emb.columns if all_data_emb[c].dtype=='object']; num_cols_emb = [c for c in all_data_emb.columns if all_data_emb[c].dtype!='object']
for c in cat_cols_emb: all_data_emb[c]=LabelEncoder().fit_transform(all_data_emb[c].astype(str))
scaler_emb = StandardScaler(); all_data_emb[num_cols_emb] = scaler_emb.fit_transform(all_data_emb[num_cols_emb])
X_emb = all_data_emb.iloc[:len(train_df_orig)]

def create_embedding_nn_model(data, cat_cols, num_cols):
    inputs, embeddings = [], [];
    for col in cat_cols:
        num_unique = int(data[col].nunique()); emb_size = int(min(np.ceil(num_unique/2), 50))
        cat_input = Input(shape=(1,), name=f'input_{col}'); emb_layer = Embedding(num_unique, emb_size)(cat_input); emb_layer = Dropout(0.3)(emb_layer)
        inputs.append(cat_input); embeddings.append(emb_layer)
    flat_embs = concatenate([Flatten()(emb) for emb in embeddings]); num_input = Input(shape=(len(num_cols),)); num_layer = BatchNormalization()(num_input)
    inputs.append(num_input); merged = concatenate([flat_embs, num_layer])
    x = Dense(512, activation=gelu)(merged); x = BatchNormalization()(x); x = Dropout(0.5)(x); x = Dense(256, activation=gelu)(x); x = BatchNormalization()(x); x = Dropout(0.4)(x)
    output = Dense(1)(x); model = Model(inputs=inputs, outputs=output); model.compile(optimizer='adam', loss='mean_squared_error'); return model

nn_emb_scores, emb_epoch_counts = [], []
print("  - Running K-Fold CV for Embedding NN... ")
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_emb, y)):
    X_train_list = [X_emb.iloc[train_idx][c].values for c in cat_cols_emb] + [X_emb.iloc[train_idx][num_cols_emb].values]
    X_val_list = [X_emb.iloc[val_idx][c].values for c in cat_cols_emb] + [X_emb.iloc[val_idx][num_cols_emb].values]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = create_embedding_nn_model(all_data_emb, cat_cols_emb, num_cols_emb)
    callbacks = [EarlyStopping(patience=40, restore_best_weights=True), LearningRateScheduler(cyclical_lr)]
    history = model.fit(X_train_list, y_train, validation_data=(X_val_list, y_val), epochs=300, batch_size=64, callbacks=callbacks, verbose=0)
    emb_epoch_counts.append(len(history.history['loss']))
    preds = model.predict(X_val_list).flatten(); score = np.sqrt(mean_squared_error(y_val, preds)); nn_emb_scores.append(score)
model_scores['EmbeddingNN'] = {'mean': np.mean(nn_emb_scores), 'std': np.std(nn_emb_scores)}
print(f"    EmbeddingNN CV RMSLE: {model_scores['EmbeddingNN']['mean']:.4f} (+/- {model_scores['EmbeddingNN']['std']:.4f})")

print("\n[Part 4/6] Evaluating Wide & Deep Neural Network...")

deep_cols = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'TotalSF', 'TotalBath', 'TotalPorchSF', 'HouseAge', 'RemodAge']
wide_cols = [col for col in X_featured.columns if col not in deep_cols]
X_wide = X_featured[wide_cols].astype(int); X_deep = X_featured[deep_cols]
scaler_wd = StandardScaler(); X_deep_scaled = scaler_wd.fit_transform(X_deep)

def create_wide_deep_model(num_wide, num_deep):
    wide_input = Input(shape=(num_wide,)); deep_input = Input(shape=(num_deep,))
    x = BatchNormalization()(deep_input); x = Dense(256, activation=gelu)(x)
    x = Dropout(0.5)(x); x = Dense(128, activation=gelu)(x); deep_output = Dropout(0.4)(x)
    merged = concatenate([wide_input, deep_output]); final_output = Dense(1)(merged)
    model = Model(inputs=[wide_input, deep_input], outputs=final_output)
    model.compile(optimizer='adam', loss='mean_squared_error'); return model

wd_scores, wd_epoch_counts = [], []
print("  - Running K-Fold CV for Wide & Deep NN... ")
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_wide)):
    X_train_wide, X_val_wide = X_wide.iloc[train_idx], X_wide.iloc[val_idx]
    X_train_deep, X_val_deep = X_deep_scaled[train_idx], X_deep_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = create_wide_deep_model(X_train_wide.shape[1], X_train_deep.shape[1])
    callbacks = [EarlyStopping(patience=40, restore_best_weights=True), LearningRateScheduler(cyclical_lr)]
    history = model.fit([X_train_wide, X_train_deep], y_train, validation_data=([X_val_wide, X_val_deep], y_val), epochs=300, batch_size=32, verbose=0, callbacks=callbacks)
    wd_epoch_counts.append(len(history.history['loss']))
    preds = model.predict([X_val_wide, X_val_deep]).flatten(); score = np.sqrt(mean_squared_error(y_val, preds)); wd_scores.append(score)
model_scores['WideDeepNN'] = {'mean': np.mean(wd_scores), 'std': np.std(wd_scores)}
print(f"    WideDeepNN CV RMSLE: {model_scores['WideDeepNN']['mean']:.4f} (+/- {model_scores['WideDeepNN']['std']:.4f})")

print("\n[Part 5/6] Generating final comparative analysis...")
comparison_df = pd.DataFrame({
    'Model': ['Baseline (Ridge)'] + list(model_scores.keys()),
    'Mean CV RMSLE': [baseline_ridge_score] + [s['mean'] for s in model_scores.values()],
    'Std Dev': [baseline_ridge_std] + [s['std'] for s in model_scores.values()]
})
comparison_df = comparison_df.sort_values(by='Mean CV RMSLE', ascending=True).reset_index(drop=True)

print("\n--- Final Model Performance Comparison ---")
print(comparison_df.to_string(index=False))

print("\nGenerating and saving comparison plots...")

plt.figure(figsize=(12, 8))
sns.barplot(x='Mean CV RMSLE', y='Model', data=comparison_df, orient='h', palette='viridis')
plt.title('Model Performance Comparison (Mean CV RMSLE)', fontsize=16)
plt.xlabel('Root Mean Squared Log Error (Lower is Better)', fontsize=12)
plt.ylabel('Model', fontsize=12)

for i, v in enumerate(comparison_df['Mean CV RMSLE']):
    plt.text(v, i, f' {v:.4f}', va='center')
plt.tight_layout()
plt.savefig('plots/model_performance_comparison.png')
print("  - Saved 'model_performance_comparison.png'")

plt.figure(figsize=(12, 8))

comparison_df_std = comparison_df.sort_values(by='Std Dev', ascending=True)
sns.barplot(x='Std Dev', y='Model', data=comparison_df_std, orient='h', palette='plasma')
plt.title('Model Stability Comparison (Standard Deviation of CV Scores)', fontsize=16)
plt.xlabel('Standard Deviation (Lower is More Stable)', fontsize=12)
plt.ylabel('Model', fontsize=12)

for index, row in comparison_df_std.iterrows():
    value = row['Std Dev']
    
    position = list(comparison_df_std.index).index(index)
    plt.text(value, position, f' {value:.4f}', va='center')


plt.tight_layout()
plt.savefig('plots/model_stability_comparison.png')
print("  - Saved 'model_stability_comparison.png'")


print("\n[Part 6/6] Training all models on full data and creating submission files...")

for name, info in traditional_models.items():
    print(f"  - Fitting and predicting with {name}...")
    data_to_fit = X_scaled_trad if info['data'] == 'scaled' else X_featured
    data_to_predict = X_test_scaled_trad if info['data'] == 'scaled' else X_test_featured
    info['model'].fit(data_to_fit, y)
    preds = np.expm1(info['model'].predict(data_to_predict))
    pd.DataFrame({'Id': test_ids, 'SalePrice': preds}).to_csv(f'submissions/submission_{name}.csv', index=False)
    print(f"     Submission file 'submissions/submission_{name}.csv' created.")

print("  - Fitting and predicting with Embedding NN...")
final_epochs_emb = max(15, int(np.mean(emb_epoch_counts) - 40))
print(f"    Training final Embedding NN for {final_epochs_emb} epochs...")
X_test_emb = all_data_emb.iloc[len(train_df_orig):]
X_list_full = [X_emb[c].values for c in cat_cols_emb] + [X_emb[num_cols_emb].values]
X_test_list = [X_test_emb[c].values for c in cat_cols_emb] + [X_test_emb[num_cols_emb].values]
final_emb_model = create_embedding_nn_model(all_data_emb, cat_cols_emb, num_cols_emb)
final_emb_model.fit(X_list_full, y, epochs=final_epochs_emb, batch_size=64, verbose=0)
preds_log = final_emb_model.predict(X_test_list).flatten(); preds = np.expm1(preds_log)
pd.DataFrame({'Id': test_ids, 'SalePrice': preds}).to_csv('submissions/submission_EmbeddingNN.csv', index=False)
print(f"     Submission file 'submissions/submission_EmbeddingNN.csv' created.")

print("  - Fitting and predicting with Wide & Deep NN...")
final_epochs_wd = max(15, int(np.mean(wd_epoch_counts) - 40))
print(f"    Training final Wide & Deep NN for {final_epochs_wd} epochs...")
X_test_wide = X_test_featured[wide_cols].astype(int); X_test_deep = X_test_featured[deep_cols]
X_test_deep_scaled = scaler_wd.transform(X_test_deep)
final_wd_model = create_wide_deep_model(X_wide.shape[1], X_deep_scaled.shape[1])
final_wd_model.fit([X_wide, X_deep_scaled], y, epochs=final_epochs_wd, batch_size=32, verbose=0)
preds_log = final_wd_model.predict([X_test_wide, X_test_deep_scaled]).flatten(); preds = np.expm1(preds_log)
pd.DataFrame({'Id': test_ids, 'SalePrice': preds}).to_csv('submissions/submission_WideDeepNN.csv', index=False)
print(f"     Submission file 'submissions/submission_WideDeepNN.csv' created.")

print("\n\n--- WEEK 4 COMPLETE ---")

