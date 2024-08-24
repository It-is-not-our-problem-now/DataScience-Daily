import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def describe_data(df):
    """
    Print descriptive statistics of the dataframe.
    """
    print("Dataset Info:")
    print(df.info())
    print("\nStatistical Summary:")
    print(df.describe())

def plot_distributions(df):
    """
    Plot distribution of numeric features.
    """
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns].hist(figsize=(15, 10), bins=20, edgecolor='black')
    plt.suptitle('Distribution of Numeric Features')
    plt.show()

def plot_correlations(df):
    """
    Plot a heatmap of feature correlations.
    """
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.show()

def identify_outliers(df):
    """
    Identify outliers using the IQR method.
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    print("Number of Outliers in Each Feature:")
    print(outliers[outliers > 0])

def main(file_path):
    df = load_data(file_path)
    describe_data(df)
    plot_distributions(df)
    plot_correlations(df)
    identify_outliers(df)

# Example usage
# Replace 'your_dataset.csv' with the path to your dataset
if __name__ == "__main__":
    main('your_dataset.csv')
