import pandas as pd
import numpy as np
import pickle
from ipywidgets import interact, fixed
import ipywidgets as widget
import random
import os

from sklearn.model_selection import cross_validate

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
custom_colors = ["#ffd670","#70d6ff","#ff4d6d","#8338ec","#90cf8e"]
customPalette = sns.set_palette(sns.color_palette(custom_colors))
sns.palplot(sns.color_palette(custom_colors),size=1.2)

def set_seed(seed):
    """
    Sets a global random seed of your choice
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def get_feats(data):
    """
    takes a dataframe as an argument and returns lists of:
    categorical, integer, continuous, null and not null columns
    """
    cat_feats = data.select_dtypes('object').columns
    int_feats = data.select_dtypes('integer').columns
    cont_feats = data.select_dtypes('float').columns
    null_feats = [col for col in data.columns if data[col].isnull().sum() != 0]
    no_nulls = [col for col in data.columns if col not in null_feats and col!='failure']
    return cat_feats, int_feats, cont_feats, null_feats, no_nulls

def get_products(df1, df2):
    """
    Concats train and test data and returns a dataframe for each product code
    """
    df = pd.concat([df1, df2], axis=0)
    dfa = df[df['product_code'] == 'A']
    dfb = df[df['product_code'] == 'B']
    dfc = df[df['product_code'] == 'C']
    dfd = df[df['product_code'] == 'D']
    dfe = df[df['product_code'] == 'E']
    dff = df[df['product_code'] == 'F']
    dfg = df[df['product_code'] == 'G']
    dfh = df[df['product_code'] == 'H']
    dfi = df[df['product_code'] == 'I']
    return dfa, dfb, dfc, dfd, dfe, dff, dfg, dfh, dfi

def get_test_product(data):
    """
    takes the test dataset and returns a list of dataframes split by product code
    """
    dff = data[data['product_code'] == 'F']
    dfg = data[data['product_code'] == 'G']
    dfh = data[data['product_code'] == 'H']
    dfi = data[data['product_code'] == 'I']
    return dff, dfg, dfh, dfi

def score(X, y, model, cv):
     scoring=['roc_auc']
     scores =cross_validate(model, X, y, scoring=scoring, cv=cv, return_train_score=True)
     scores = pd.DataFrame(scores).T
     return scores.assign(
         mean=lambda x: x.mean(axis=1), 
         std= lambda x: x.std(axis=1))

def descriptive_info_test(data, dtype):
    """
    This function is meant to be used in tandem with the decorator:
    '@interact(data=fixed(df), dtype=['integer', 'float', 'object'])'

    Arguments:
    Train or test dataframe,
    and your chosen datatype if the interact widget is not used.

    Returns:
    A dataframe containing everything the method 'describe()' does as well as the
    additional columns: dtype, missing_values, unique values and missing value percent
    """
    dt = data.select_dtypes(dtype).columns
    info = pd.DataFrame({
        'dtype': data[dt].dtypes.tolist(),
        'missing_values': data[dt].isnull().sum().values,
        'unique_values': data[dt].nunique().values}, 
        index = data[dt].columns.values)
    info['missing_value_pct'] = (info['missing_values'] / data.shape[0]) * 100
    description = data[dt].describe().T
    return pd.concat([description, info], axis=1)

def clf_plot_distributions(
    train_df, test_df, features, save_filepath,
    suptitle='Density Histograms of Continuous Features for Train and Test Datasets with the Probability of Failure Plotted on the Secondary y-Axis', 
    ncols=4
    ):
    """
    This function plots histograms of your chosen features for training and test datasets as well as target probabilities. 
    Each plot contains overlapping histograms of the train and test feature and a scatterplot of the target on the secondary y-axis.

    Arguments:
    train_df, test_df, a list of features for the histograms,
    the filepath to save the image to,
    a super-title string
    and the number of columns youd like to have

    Returns:
    An assortment of plots, and a saved png file
    """
    sns.set_style(style='white')
    nrows = int(len(features) / ncols) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, round(nrows*12/ncols)))
    for ax,feature in zip(axes.ravel()[:len(features)],features):
        maximum = max(train_df[feature].max(), test_df[feature].max())
        minimum = min(train_df[feature].min(), test_df[feature].min())
        bins = np.linspace(minimum, maximum, 40)
        sns.distplot(train_df[feature], ax=ax, bins=bins, label='Train', hist_kws={'histtype':'stepfilled', 'alpha': 0.6}, kde=False, norm_hist=True)
        sns.distplot(test_df[feature], ax=ax, bins=bins, label='Test', hist_kws={'histtype':'stepfilled', 'alpha': 0.6}, kde=False, norm_hist=True)
            
        ax2 = ax.twinx()
        total, _ = np.histogram(train_df[feature], bins=bins)
        fails, _ = np.histogram(train_df[feature][train_df['failure'] == 1], bins=bins)
        fail_prob = np.nan_to_num((fails/total), nan=0)
        sns.scatterplot(x=(bins[1:] + bins[:-1])/2, y=fail_prob, label='Probability of Failure', ax=ax2, color=custom_colors[-2])
        ax2.set_ylim(0, 0.6)
        if ax == axes[0, 0]: 
            ax.legend(loc='lower right')
        else:
            leg = ax2.get_legend()
            leg.remove()
        # if ax2 == axes[0, 0]: ax2.legend(loc='upper right')
    for ax in axes.ravel()[len(features):]:
        ax.set_visible(False)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(suptitle)
    plt.show()
    plt.savefig(save_filepath)