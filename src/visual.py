import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_churn_by_categories(df, column_names, target_col='Exited', n_cols=2, figsize=(14, 5)):
    n = len(column_names)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
    axes = np.array(axes).reshape(-1) 

    total = len(df)
    for ax, col in zip(axes, column_names):
        sns.countplot(x=col, hue=target_col, data=df, palette='viridis', ax=ax)
        ax.set_title(f'{col} -> {target_col}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Müşteri Sayısı', fontsize=10)
        ax.legend(title='Churn', labels=['Kaldı (0)', 'Gitti (1)'])

        for p in ax.patches:
            height = p.get_height()
            if pd.isna(height):
                height = 0
            percentage = f'{100 * height / total:.1f}%'
            ax.annotate(f'{int(height)}\n({percentage})',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center',
                        xytext=(0, 7),
                        textcoords='offset points',
                        fontsize=9)

    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, figsize=(12, 8)):

    plt.figure(figsize=figsize)
    
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    heatmap = sns.heatmap(numeric_df.corr(), 
                          annot=True, 
                          cmap='coolwarm', 
                          fmt=".2f", 
                          linewidths=.5)
    
    plt.title('Değişkenler Arası Korelasyon Haritası', fontsize=16)
    plt.show()

def plot_boxplot(df, column_name, figsize=(8, 4)):
    
    plt.figure(figsize=figsize)
    
    sns.boxplot(x=df[column_name], color='skyblue')
    
    plt.title(f'{column_name} Değişkeni İçin Outlier Analizi', fontsize=12, fontweight='bold')
    plt.xlabel(column_name, fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def plot_multi_boxplot(df, column_names, figsize=(12, 4)):

    n_cols = len(column_names)
    
    fig, axes = plt.subplots(1, n_cols, figsize=figsize, sharey=False)
    
    if n_cols == 1:
        axes = [axes]
    
    for ax, column_name in zip(axes, column_names):
        sns.boxplot(y=df[column_name], ax=ax, color='skyblue')
        ax.set_title(f'{column_name} Outlier Analizi', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(column_name, fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()


def outlier_thresholds(dataframe, variable, low_quantile=0.05, up_quantile=0.95):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


def has_outliers(dataframe, numeric_columns, plot=False):
    for col in numeric_columns:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col, " : ", number_of_outliers, "outliers")
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()
