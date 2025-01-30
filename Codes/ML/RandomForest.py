#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

# Load the dataset
df = pd.read_csv('ML_lists.csv')

# Prepare the feature set by dropping unnecessary columns for MD analysis
X = df.drop(columns=['Polymer Name','LL_embd'])  # drop SCP names and target property
y = df['LL_embd']  # Target variable

random_states = [52, 0, 41, 65, 100]  # List of random states for reproducibility

# To store cumulative distributions and results
cumulative_y_test = []
cumulative_y_pred = []
mse_list = []
cross_val_mse_list = []
spearman_list = []

# Loop over random states for multiple train-test splits
for random_state in random_states:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    
    # Scale the features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Sort true and predicted values for cumulative distribution
    sorted_indices = np.argsort(y_test)
    cumulative_y_test.append(y_test.iloc[sorted_indices].values)
    cumulative_y_pred.append(y_pred[sorted_indices])
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)
    
    # Cross-validation MSE
    cross_val_mse = -cross_val_score(
        model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error'
    ).mean()
    cross_val_mse_list.append(cross_val_mse)
    
    # Spearman correlation between true and predicted values
    spearman_corr = spearmanr(y_test, y_pred).correlation
    spearman_list.append(spearman_corr)

# Compute average cumulative distributions
mean_y_test = np.mean(cumulative_y_test, axis=0)
mean_y_pred = np.mean(cumulative_y_pred, axis=0)

# Calculate average metrics
avg_rmse = np.sqrt(np.mean(mse_list))
avg_cross_val_rmse = np.sqrt(np.mean(cross_val_mse_list))
avg_spearman = np.mean(spearman_list)

# Print results
print(f"Average RMSE: {avg_rmse:.4f}")
print(f"Average Cross-validation RMSE: {avg_cross_val_rmse:.4f}")
print(f"Average Spearman Correlation: {avg_spearman:.4f}")




# Function to create a three-shaded scatter plot with trendline
def three_shaded_scatter_with_line(x, y, rank_x, rank_y, ax, size=0.5):
    """
    Create a scatter plot with three-shaded markers and a trendline.

    Args:
        x (array-like): X-axis values (y_test).
        y (array-like): Y-axis values (predictions).
        rank_x (array-like): Ranks of x values.
        rank_y (array-like): Ranks of y values.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        size (float): Size of the half-circle patches.
    """
    max_val = max(x)  # Set axis limits to match the range of y_test
    ax.set_xlim(5, 37)
    ax.set_ylim(5, 37)
    
    # Plot the trendline using Seaborn's regplot
    sns.regplot(x=x, y=y, scatter=False, line_kws={"color": "black", "lw": 2}, ci=None, ax=ax)
    
    # Plot three-shaded markers
    for xi, yi, rx, ry in zip(x, y, rank_x, rank_y):
        # Determine colors based on thirds of ranks
        if rx < len(rank_x) / 3:
            color_top = "navy"
        elif rx < 2 * len(rank_x) / 3:
            color_top = "olive"
        else:
            color_top = "crimson"

        if ry < len(rank_y) / 3:
            color_bottom = "navy"
        elif ry < 2 * len(rank_y) / 3:
            color_bottom = "olive"
        else:
            color_bottom = "crimson"

        # Create half-circle patches for each data point
        circle_top = mpatches.Wedge((xi, yi), size, theta1=0, theta2=180, color=color_top)
        circle_bottom = mpatches.Wedge((xi, yi), size, theta1=180, theta2=360, color=color_bottom)
        ax.add_patch(circle_top)
        ax.add_patch(circle_bottom)

# Add a custom legend to the plot
def add_custom_legend(ax):
    
    # Create custom handles for the legend
    navy_patch = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='navy', markersize=10, label='Bottom Third')
    olive_patch = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='olive', markersize=10, label='Middle Third')
    crimson_patch = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='crimson', markersize=10, label='Top Third')

    # Add the legend to the axes
    ax.legend(handles=[navy_patch, olive_patch, crimson_patch], loc='upper left', fontsize=10, frameon=False)

# Calculate ranks for y_test and y_pred_md
rank_test = np.argsort(np.argsort(y_test))  # Rank of y_test
rank_pred_md = np.argsort(np.argsort(y_pred))  # Rank of y_pred_md

# Create a scatter plot with half-shaded markers and trendline (Figure 4 of the paper)
fig, ax = plt.subplots(figsize=(5, 5))
three_shaded_scatter_with_line(y_test, y_pred, rank_test, rank_pred_md, ax, size=0.5)
spearman_corr_md = spearmanr(y_test, y_pred).correlation  # Calculate Spearman correlation for MD predictions
ax.set_xlabel('LL(embd) / Å')
ax.set_ylabel('LL(ML) / Å')
add_custom_legend(ax)  # Add custom legend
plt.tight_layout()  # Adjust layout
plt.savefig('ranking.png')  # Save the plot as an EPS file
plt.show()  # Display the plot


'''
FEATURE IMPORTANCE
'''
# After fitting the model with x and y, get feature importance
feature_importance = model.feature_importances_

# Define the positions for each feature
indices = np.arange(len(feature_importance))

# Plot feature importance
plt.barh(indices, feature_importance, color='steelblue')

# Define custom names for the y-ticks
custom_feature_names = [r'$\mathrm{L}_{pp}$', r'$\mu^{GAS}$', r'$\beta$'] 


plt.yticks(indices, custom_feature_names)


plt.xlabel('Feature Importance')


plt.tight_layout()
plt.savefig('md_feature_importance.png')
plt.show()


# In[ ]:





# In[ ]:




