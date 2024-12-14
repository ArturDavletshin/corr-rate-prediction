# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:29:57 2024

@author: Artur
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:24:20 2024

@author: Artur
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor  # Import XGBoost
import shap
import seaborn as sns
import matplotlib.pyplot as plt

# --- Step 1: Load the Dataset ---
file_path = 'database.csv'
df = pd.read_csv(file_path)

# Drop unnecessary columns
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
if 'Alloy' in df.columns:
    print("\nDropping 'Alloy' column...")
    df.drop(columns=['Alloy'], inplace=True)
if 'phase ' in df.columns:
    print("\nDropping 'phase' column...")
    df.drop(columns=['phase '], inplace=True)

# Replace '-' with NaN
df.replace('-', np.nan, inplace=True)

# --- Step 2: Handle Missing Values ---
print("\nImputing Missing Values...")
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# --- Step 3: Encode Categorical Data ---
non_numeric_columns = df_imputed.select_dtypes(exclude=[np.number]).columns
print("Non-Numeric Columns:", list(non_numeric_columns))

if not non_numeric_columns.empty:
    df_imputed = pd.get_dummies(df_imputed, columns=non_numeric_columns, drop_first=True)

# --- Step 4: Select Features and Target ---
target_column = 'corrosion rate CR~ 0.05-0.075 mm/year'

if target_column not in df_imputed.columns:
    raise ValueError(f"Target column '{target_column}' not found in the dataset.")

X = df_imputed.drop(columns=[target_column])
y = df_imputed[target_column]

# Standardize features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Feature Selection
k = 5  # Adjust the number of features to select
selector = SelectKBest(score_func=f_regression, k=k)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = X_scaled.columns[selector.get_support()]
print("\nSelected Features:", list(selected_features))

from sklearn.manifold import TSNE

# --- Calculate t-SNE Embeddings ---
print("\nCalculating t-SNE Embeddings...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_features = tsne.fit_transform(X_scaled)  # Compute t-SNE on the scaled dataset

# Add t-SNE features to the scaled dataset
X_scaled['t-SNE-1'] = tsne_features[:, 0]
X_scaled['t-SNE-2'] = tsne_features[:, 1]

# Update the feature selection step to include t-SNE features
X_selected = X_scaled[selected_features.tolist() + ['t-SNE-1', 't-SNE-2']]
print("\nt-SNE Features Added: ['t-SNE-1', 't-SNE-2']")
print(f"New Feature Set Size: {X_selected.shape}")








# --- Step 5: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# --- Step 6: Train and Evaluate Models ---
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Train and evaluate a model using train-test split.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    return model, r2

# Add XGBoost to the model list
models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42)
}

best_r2 = -np.inf
best_model = None

print("\n--- Evaluating Models ---")
for name, model in models.items():
    trained_model, r2 = evaluate_model(model, X_train, y_train, X_test, y_test)
    if r2 > best_r2:
        best_r2 = r2
        best_model = trained_model

print(f"\nBest Model: {best_model.__class__.__name__} with R^2: {best_r2:.4f}")

# --- Iteratively Remove High-Residual Rows with Limit ---
def iterative_remove_high_residuals_with_limit(model, X, y, threshold_percentile=90, max_row_loss=None):
    """
    Iteratively removes rows with high residuals based on a threshold percentile.
    Stops if max_row_loss is exceeded.
    """
    iteration = 0
    total_rows = len(y)
    rows_removed = 0

    while True:
        model.fit(X, y)
        residuals = np.abs(y - model.predict(X))
        threshold = np.percentile(residuals, threshold_percentile)
        high_residual_rows = residuals > threshold
        num_rows_to_remove = high_residual_rows.sum()

        if num_rows_to_remove == 0:
            print("\nNo more high-residual rows to remove.")
            break

        # Check if removing these rows would exceed max_row_loss
        if max_row_loss is not None and rows_removed + num_rows_to_remove > max_row_loss:
            print(f"\nStopping to avoid exceeding max row loss of {max_row_loss}.")
            break

        print(f"Iteration {iteration + 1}: Removing {num_rows_to_remove} high-residual rows...")
        X = X[~high_residual_rows]
        y = y[~high_residual_rows]
        rows_removed += num_rows_to_remove
        iteration += 1

    print(f"\nTotal Rows Removed: {rows_removed} (out of {total_rows})")
    return X, y

# Apply to training data with a row loss limit
max_loss_limit = 10 # Set maximum rows you are willing to lose
print("\n--- Removing High-Residual Rows with Limit ---")
X_train_cleaned, y_train_cleaned = iterative_remove_high_residuals_with_limit(
    best_model, X_train, y_train, threshold_percentile=90, max_row_loss=max_loss_limit
)

print(f"\nCleaned Training Data Size: {X_train_cleaned.shape[0]} rows, {X_train_cleaned.shape[1]} features")

# --- Re-Evaluate Models After Cleaning ---
print("\n--- Evaluating Models on Cleaned Data ---")
for name, model in models.items():
    trained_model, r2 = evaluate_model(model, X_train_cleaned, y_train_cleaned, X_test, y_test)
    if r2 > best_r2:
        best_r2 = r2
        best_model = trained_model

print(f"\nBest Model after Cleaning: {best_model.__class__.__name__} with R^2: {best_r2:.4f}")

# Collect and rank models by their R^2 scores
model_scores = []

print("\n--- Evaluating and Ranking Models ---")
for name, model in models.items():
    trained_model, r2 = evaluate_model(model, X_train_cleaned, y_train_cleaned, X_test, y_test)
    model_scores.append((name, trained_model, r2))

# Sort models by R^2 score in descending order
model_scores = sorted(model_scores, key=lambda x: x[2], reverse=True)

# Print the top 5 models
print("\n--- Top 5 Models ---")
for rank, (name, model, r2) in enumerate(model_scores[:5], start=1):
    print(f"{rank}. {name}: R^2 = {r2:.4f}")

#####################################################until here its fine!





#####################################################
'''
# --- Analyzing R^2 Scores for Varying max_loss_limit ---
loss_limits = list(range(0, 200, 5))  # From 0 to 150, step 5
r2_scores_rf = []  # To store R^2 scores for Random Forest
r2_scores_GB = []  # To store R^2 scores for Ridge Regression
r2_scores_xgb = []  # To store R^2 scores for XGBoost

print("\n--- Evaluating Models for Different max_loss_limit Values ---")

for max_loss in loss_limits:
    # Clean data based on max_loss_limit
    X_train_cleaned, y_train_cleaned = iterative_remove_high_residuals_with_limit(
        best_model, X_train, y_train, threshold_percentile=90, max_row_loss=max_loss
    )
    
    # Evaluate Random Forest
    rf = RandomForestRegressor(random_state=42)
    _, r2_rf = evaluate_model(rf, X_train_cleaned, y_train_cleaned, X_test, y_test)
    r2_scores_rf.append(r2_rf)
    
    # Evaluate Ridge Regression
    rr = GradientBoostingRegressor(random_state=42)
    _, r2_rr = evaluate_model(rr, X_train_cleaned, y_train_cleaned, X_test, y_test)
    r2_scores_GB.append(r2_rr)
    
    # Evaluate XGBoost
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
    _, r2_xgb = evaluate_model(xgb, X_train_cleaned, y_train_cleaned, X_test, y_test)
    r2_scores_xgb.append(r2_xgb)

# Plot the R^2 scores
plt.figure(figsize=(10, 6))
plt.plot(loss_limits, r2_scores_rf, label="Random Forest", marker='o')
plt.plot(loss_limits, r2_scores_GB, label="Gradient Boosting", marker='o')
plt.plot(loss_limits, r2_scores_xgb, label="XGBoost", marker='o')
plt.title(r"$R^2$ Scores for Varying maximum loss limit (Number of measurements)")
plt.xlabel("Maximum loss limit")
plt.ylabel(r"$R^2$ Score")
plt.legend()
plt.grid(False)
plt.show()
'''

##########################################################tuning
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# --- Fine-Tuning Hyperparameters for Top 3 Models ---

# 1. Fine-tuning XGBoost
xgb_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

print("\n--- Fine-Tuning XGBoost ---")
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=xgb_param_grid,
    scoring='r2',
    cv=3,
    verbose=2,
    n_jobs=-1
)
xgb_grid_search.fit(X_train_cleaned, y_train_cleaned)
print(f"Best Parameters for XGBoost: {xgb_grid_search.best_params_}")
print(f"Best R^2 Score for XGBoost: {xgb_grid_search.best_score_:.4f}")

# 2. Fine-tuning Gradient Boosting
gb_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

print("\n--- Fine-Tuning Gradient Boosting ---")
gb = GradientBoostingRegressor(random_state=42)
gb_grid_search = GridSearchCV(
    estimator=gb,
    param_grid=gb_param_grid,
    scoring='r2',
    cv=3,
    verbose=2,
    n_jobs=-1
)
gb_grid_search.fit(X_train_cleaned, y_train_cleaned)
print(f"Best Parameters for Gradient Boosting: {gb_grid_search.best_params_}")
print(f"Best R^2 Score for Gradient Boosting: {gb_grid_search.best_score_:.4f}")

# 3. Fine-tuning Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("\n--- Fine-Tuning Random Forest ---")
rf = RandomForestRegressor(random_state=42)
rf_grid_search = GridSearchCV(
    estimator=rf,
    param_grid=rf_param_grid,
    scoring='r2',
    cv=3,
    verbose=2,
    n_jobs=-1
)
rf_grid_search.fit(X_train_cleaned, y_train_cleaned)
print(f"Best Parameters for Random Forest: {rf_grid_search.best_params_}")
print(f"Best R^2 Score for Random Forest: {rf_grid_search.best_score_:.4f}")

# --- Evaluate Fine-Tuned Models on Test Set ---
print("\n--- Evaluating Fine-Tuned Models on Test Set ---")

# Evaluate XGBoost
best_xgb = xgb_grid_search.best_estimator_
_, r2_xgb_test = evaluate_model(best_xgb, X_train_cleaned, y_train_cleaned, X_test, y_test)
print(f"XGBoost Test R^2 Score: {r2_xgb_test:.4f}")

# Evaluate Gradient Boosting
best_gb = gb_grid_search.best_estimator_
_, r2_gb_test = evaluate_model(best_gb, X_train_cleaned, y_train_cleaned, X_test, y_test)
print(f"Gradient Boosting Test R^2 Score: {r2_gb_test:.4f}")

# Evaluate Random Forest
best_rf = rf_grid_search.best_estimator_
_, r2_rf_test = evaluate_model(best_rf, X_train_cleaned, y_train_cleaned, X_test, y_test)
print(f"Random Forest Test R^2 Score: {r2_rf_test:.4f}")

# --- Visualize Results ---
model_names = ['XGBoost', 'Gradient Boosting', 'Random Forest']
r2_scores = [r2_xgb_test, r2_gb_test, r2_rf_test]

plt.figure(figsize=(10, 6))
plt.bar(model_names, r2_scores, color=['blue', 'green', 'orange'])
plt.title('R² Scores of Fine-Tuned Models')
plt.ylabel('R² Score')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


######################################
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- Evaluate Fine-Tuned Models on Test Set and Store Metrics ---
print("\n--- Evaluating Fine-Tuned Models on Test Set ---")

# Helper function to evaluate and collect metrics
def evaluate_model_with_metrics(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    
    return r2, mae, mape

# Initialize lists to store metrics
model_names = ['XGBoost', 'Gradient Boosting', 'Random Forest']
r2_scores, mae_scores, mape_scores = [], [], []

# Evaluate XGBoost
best_xgb = xgb_grid_search.best_estimator_
r2_xgb, mae_xgb, mape_xgb = evaluate_model_with_metrics(best_xgb, X_train_cleaned, y_train_cleaned, X_test, y_test)
r2_scores.append(r2_xgb)
mae_scores.append(mae_xgb)
mape_scores.append(mape_xgb)

# Evaluate Gradient Boosting
best_gb = gb_grid_search.best_estimator_
r2_gb, mae_gb, mape_gb = evaluate_model_with_metrics(best_gb, X_train_cleaned, y_train_cleaned, X_test, y_test)
r2_scores.append(r2_gb)
mae_scores.append(mae_gb)
mape_scores.append(mape_gb)

# Evaluate Random Forest
best_rf = rf_grid_search.best_estimator_
r2_rf, mae_rf, mape_rf = evaluate_model_with_metrics(best_rf, X_train_cleaned, y_train_cleaned, X_test, y_test)
r2_scores.append(r2_rf)
mae_scores.append(mae_rf)
mape_scores.append(mape_rf)

########################################################################################3tuning parameters
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# --- Evaluate Fine-Tuned Models on Test Set and Store Metrics ---
print("\n--- Evaluating Fine-Tuned Models on Test Set ---")

# Helper function to evaluate and collect metrics
def evaluate_model_with_metrics(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return r2, mae, mape

# Initialize lists to store metrics
model_names = ['XGBoost', 'Gradient Boosting', 'Random Forest']
r2_scores, mae_scores, mape_scores = [], [], []
parameters = []

# Evaluate XGBoost
best_xgb = xgb_grid_search.best_estimator_
r2_xgb, mae_xgb, mape_xgb = evaluate_model_with_metrics(best_xgb, X_train_cleaned, y_train_cleaned, X_test, y_test)
r2_scores.append(r2_xgb)
mae_scores.append(mae_xgb)
mape_scores.append(mape_xgb)
parameters.append(best_xgb.get_params())

# Evaluate Gradient Boosting
best_gb = gb_grid_search.best_estimator_
r2_gb, mae_gb, mape_gb = evaluate_model_with_metrics(best_gb, X_train_cleaned, y_train_cleaned, X_test, y_test)
r2_scores.append(r2_gb)
mae_scores.append(mae_gb)
mape_scores.append(mape_gb)
parameters.append(best_gb.get_params())

# Evaluate Random Forest
best_rf = rf_grid_search.best_estimator_
r2_rf, mae_rf, mape_rf = evaluate_model_with_metrics(best_rf, X_train_cleaned, y_train_cleaned, X_test, y_test)
r2_scores.append(r2_rf)
mae_scores.append(mae_rf)
mape_scores.append(mape_rf)
parameters.append(best_rf.get_params())

# Create a DataFrame to store the results
results = pd.DataFrame({
    "Model": model_names,
    "R²": r2_scores,
    "MAE": mae_scores,
    "MAPE": mape_scores,
    "Best Parameters": parameters
})

# Save the DataFrame to a CSV file
output_file_path = "model_evaluation_metrics.csv"
results.to_csv(output_file_path, index=False)

print(f"Model evaluation metrics and parameters saved to {output_file_path}")


# Optional: You can plot or save these results for comparison







#########################################################################################



# --- High-Quality Visualizations ---
# --- High-Quality Aesthetics ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- High-Quality Aesthetics ---
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

# --- High-Quality Aesthetics ---
plt.rcParams.update({
    'font.family': 'serif',
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'axes.linewidth': 1.5
})

# Define bar positions
bar_width = 0.4
x = np.arange(len(model_names))  # X-axis positions for R² bars

# --- Plot R² and MAE as Side-by-Side Bars ---
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot R² on the primary y-axis
color_r2 = '#1f77b4'  # Blue color
r2_bars = ax1.bar(x - bar_width / 2, r2_scores, bar_width, color=color_r2, alpha=0.8, label='R²')
ax1.set_ylabel('R² Score', fontsize=20, color=color_r2)
ax1.tick_params(axis='y', labelcolor=color_r2, labelsize=20)  # Set font size for axis values
ax1.set_ylim(0, 1.1)  # Extend y-limit for R²
for bar, score in zip(r2_bars, r2_scores):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{score:.2f}', 
             ha='center', va='bottom', fontsize=20, color=color_r2)

# Create a secondary y-axis for MAE
color_mae = '#2ca02c'  # Green color
ax2 = ax1.twinx()
mae_bars = ax2.bar(x + bar_width / 2, mae_scores, bar_width, color=color_mae, alpha=0.8, label='MAE')
ax2.set_ylabel('Mean Absolute Error (MAE)', fontsize=20, color=color_mae)
ax2.tick_params(axis='y', labelcolor=color_mae, labelsize=20)  # Set font size for axis values
ax2.set_ylim(0, max(mae_scores) * 1.5)  # Scale MAE appropriately
for bar, score in zip(mae_bars, mae_scores):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{score:.2f}', 
             ha='center', va='bottom', fontsize=20, color=color_mae)

# Add title and adjust layout
#plt.title('R² and MAE Scores of Fine-Tuned Models', fontsize=18, weight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, fontsize=20)  # Set font size for x-axis tick labels

fig.tight_layout()

# Save and show the bar plot
plt.savefig("r2_mae_side_by_side_green_high_quality.png", dpi=300, bbox_inches='tight')
ax1.grid(False)
ax2.grid(False)
plt.show()

###############################################################################
# --- Scatter Plot for y_true vs y_pred ---
def plot_scatter(y_true, y_pred, model_name):
    """
    Plot scatter plot for true vs predicted values.
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, color='#1f77b4', alpha=0.7, edgecolors='k', label='Data Points', s=200)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linewidth=2, linestyle='--', label='Ideal Fit')
    plt.title(f'{model_name}: Predicted vs True Values', fontsize=24, weight='bold')
    plt.xlabel('True Values', fontsize=24)
    plt.ylabel('Predicted Values', fontsize=24)
    plt.tick_params(axis='both', labelsize=24)  # Increase the font size of axis tick labels
    plt.legend(fontsize=24)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"scatter_{model_name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()


# Generate scatter plots for all three models
best_models = {
    "XGBoost": best_xgb,
    "Gradient Boosting": best_gb,
    "Random Forest": best_rf
}

print("\n--- Generating Scatter Plots for True vs Predicted Values ---")
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    plot_scatter(y_test, y_pred, model_name)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#####################################################
import pandas as pd
import random

# Consolidate predictions for 5 random samples into a single table
consolidated_table = []

for i in range(5):
    random_index = random.randint(0, len(X_test) - 1)  # Random index
    for model_name, model in best_models.items():
        single_prediction = model.predict(X_test.iloc[[random_index]])[0]
        true_value = y_test.iloc[random_index]
        consolidated_table.append({
            "Random Sample Index": random_index,
            "Model": model_name,
            "True Value": true_value,
            "Predicted Value": single_prediction
        })

# Convert to DataFrame for display
consolidated_df = pd.DataFrame(consolidated_table)

# Display the consolidated table
print(consolidated_df)


