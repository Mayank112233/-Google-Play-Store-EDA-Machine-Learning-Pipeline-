import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# Create plots folder
os.makedirs('plots', exist_ok=True)
sns.set_theme(style="whitegrid")

def clean_size(size_str):
    if pd.isna(size_str) or size_str == 'Varies with device':
        return np.nan
    size_str = str(size_str).strip()
    if size_str.endswith('M') or size_str.endswith('m'):
        return float(size_str[:-1])
    elif size_str.endswith('k') or size_str.endswith('K'):
        return float(size_str[:-1]) / 1024.0
    try:
        return float(size_str)
    except:
        return np.nan

def clean_genres(genre_str):
    if pd.isna(genre_str):
        return 'Unknown', 'None'
    parts = str(genre_str).split(';')
    primary = parts[0].strip()
    secondary = parts[1].strip() if len(parts) > 1 else 'None'
    return primary, secondary

def main():
    print("--- 1. Loading & Cleaning Data ---")
    df = pd.read_csv('googleplaystore.csv')
    print(f"Initial shape: {df.shape}")

    # Remove corrupted row index 10472 (Category '1.9', Rating 19.0)
    df = df[df['Category'] != '1.9'].reset_index(drop=True)
    print(f"Shape after removing corrupt row: {df.shape}")

    # Clean Reviews
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')

    # Clean Installs
    df['Installs'] = df['Installs'].astype(str).str.replace('+', '', regex=False).str.replace(',', '', regex=False)
    df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')

    # Clean Price
    df['Price'] = df['Price'].astype(str).str.replace('$', '', regex=False)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # Clean Size
    df['Size'] = df['Size'].apply(clean_size)

    # Impute missing Size with median per Category
    df['Size'] = df.groupby('Category')['Size'].transform(lambda x: x.fillna(x.median()))
    df['Size'] = df['Size'].fillna(df['Size'].median())

    # Impute missing Rating with median per Category
    df['Rating'] = df.groupby('Category')['Rating'].transform(lambda x: x.fillna(x.median()))
    df['Rating'] = df['Rating'].fillna(df['Rating'].median())

    # Impute missing Type with mode
    mode_type = df['Type'].mode()[0]
    df['Type'] = df['Type'].fillna(mode_type)
    # Fix any misalignment where Price is 0 but Type is Paid
    df.loc[df['Price'] == 0, 'Type'] = 'Free'

    # Clean Current Ver
    mode_ver = df['Current Ver'].mode()[0]
    df['Current Ver'] = df['Current Ver'].fillna(mode_ver)
    df['Current Ver_Clean'] = df['Current Ver'].astype(str).str.extract(r'(\d+\.\d+)')[0].fillna('1.0')

    # Clean Genres into Primary and Secondary
    genres_split = df['Genres'].apply(clean_genres)
    df['Primary_Genre'] = [g[0] for g in genres_split]
    df['Secondary_Genre'] = [g[1] for g in genres_split]

    print("Missing values after cleaning:")
    print(df[['Rating', 'Size', 'Type', 'Current Ver', 'Primary_Genre']].isnull().sum())

    # Plot replaced sns.histplot for distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df['Rating'], kde=True, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Distribution of App Ratings (sns.histplot)')

    sns.histplot(df['Size'], kde=True, ax=axes[0, 1], color='salmon')
    axes[0, 1].set_title('Distribution of App Size (MB)')

    sns.histplot(np.log1p(df['Installs']), kde=True, ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title('Distribution of Log(Installs + 1)')

    sns.histplot(np.log1p(df['Reviews']), kde=True, ax=axes[1, 1], color='purple')
    axes[1, 1].set_title('Distribution of Log(Reviews + 1)')

    plt.tight_layout()
    plt.savefig('plots/1_clean_distributions.png', dpi=300)
    plt.close()

    print("\n--- 2. Handling Outliers ---")
    # Plot Boxplots before capping
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.boxplot(y=df['Rating'], ax=axes[0], color='skyblue')
    axes[0].set_title('Boxplot: Rating')

    sns.boxplot(y=df['Installs'], ax=axes[1], color='lightgreen')
    axes[1].set_title('Boxplot: Installs (Raw)')

    sns.boxplot(y=df['Reviews'], ax=axes[2], color='purple')
    axes[2].set_title('Boxplot: Reviews (Raw)')

    plt.tight_layout()
    plt.savefig('plots/2_outliers_before.png', dpi=300)
    plt.close()

    # IQR / Percentile Capping for Installs, Reviews, Price
    def cap_outliers_percentile(series, p=0.95):
        upper_bound = series.quantile(p)
        return np.clip(series, 0, upper_bound)

    df['Installs_Capped'] = cap_outliers_percentile(df['Installs'], 0.95)
    df['Reviews_Capped'] = cap_outliers_percentile(df['Reviews'], 0.95)
    df['Price_Capped'] = cap_outliers_percentile(df['Price'], 0.99)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.boxplot(y=df['Rating'], ax=axes[0], color='skyblue')
    axes[0].set_title('Boxplot: Rating (Imputed)')

    sns.boxplot(y=df['Installs_Capped'], ax=axes[1], color='lightgreen')
    axes[1].set_title('Boxplot: Installs (Capped 95th Percentile)')

    sns.boxplot(y=df['Reviews_Capped'], ax=axes[2], color='purple')
    axes[2].set_title('Boxplot: Reviews (Capped 95th Percentile)')

    plt.tight_layout()
    plt.savefig('plots/2_outliers_after.png', dpi=300)
    plt.close()

    print("\n--- 3. Correlation & Statistical Analysis ---")
    corr_cols = ['Rating', 'Reviews_Capped', 'Installs_Capped', 'Price_Capped', 'Size']
    corr_matrix = df[corr_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/3_correlation_heatmap.png', dpi=300)
    plt.close()

    # Hypothesis Testing
    # Test 1: Pearson correlation between Size and Rating
    valid_size_rating = df.dropna(subset=['Size', 'Rating'])
    pearson_corr, p_val_pearson = stats.pearsonr(valid_size_rating['Size'], valid_size_rating['Rating'])
    print(f"Hypothesis Test 1 (Pearson: Size vs Rating): r = {pearson_corr:.4f}, p-value = {p_val_pearson:.4e}")

    # Test 2: ANOVA across top 5 Categories for Rating
    top5_cats = df['Category'].value_counts().head(5).index
    cat_ratings = [df[df['Category'] == cat]['Rating'] for cat in top5_cats]
    f_stat, p_val_anova = stats.f_oneway(*cat_ratings)
    print(f"Hypothesis Test 2 (ANOVA: Rating across Top 5 Categories): F = {f_stat:.4f}, p-value = {p_val_anova:.4e}")

    # Test 3: Two-sample T-test for Free vs Paid Rating
    free_ratings = df[df['Type'] == 'Free']['Rating']
    paid_ratings = df[df['Type'] == 'Paid']['Rating']
    t_stat, p_val_ttest = stats.ttest_ind(free_ratings, paid_ratings, equal_var=False)
    print(f"Hypothesis Test 3 (T-test: Free vs Paid Ratings): t = {t_stat:.4f}, p-value = {p_val_ttest:.4e}")

    print("\n--- 4. Feature Engineering ---")
    # Installs Bucket
    bins_installs = [-1, 10000, 1000000, np.inf]
    labels_installs = ['Low', 'Medium', 'High']
    df['Installs_Bucket'] = pd.cut(df['Installs'], bins=bins_installs, labels=labels_installs)

    # Rating Category
    bins_rating = [-1, 3.5, 4.0, 4.5, 5.0]
    labels_rating = ['Poor', 'Average', 'Good', 'Excellent']
    df['Rating_Category'] = pd.cut(df['Rating'], bins=bins_rating, labels=labels_rating)

    # Revenue Potential
    df['Revenue_Potential'] = df['Installs'] * df['Price']

    # Update Recency (days since Last Updated)
    df['Last_Updated_DT'] = pd.to_datetime(df['Last Updated'], errors='coerce')
    snapshot_date = df['Last_Updated_DT'].max()
    df['Update_Recency_Days'] = (snapshot_date - df['Last_Updated_DT']).dt.days
    df['Update_Recency_Days'] = df['Update_Recency_Days'].fillna(df['Update_Recency_Days'].median())

    print("Engineered Features Sample:")
    print(df[['App', 'Installs_Bucket', 'Rating_Category', 'Revenue_Potential', 'Update_Recency_Days']].head())

    print("\n--- 5. Homework & Business Questions ---")
    # Q1: Top 5 installed categories by Type (Free/Paid)
    cat_type_installs = df.groupby(['Category', 'Type'])['Installs'].sum().reset_index()
    top5_free = cat_type_installs[cat_type_installs['Type'] == 'Free'].sort_values(by='Installs', ascending=False).head(5)
    top5_paid = cat_type_installs[cat_type_installs['Type'] == 'Paid'].sort_values(by='Installs', ascending=False).head(5)
    top5_combined = pd.concat([top5_free, top5_paid])

    plt.figure(figsize=(12, 6))
    sns.barplot(data=top5_combined, x='Installs', y='Category', hue='Type', palette={'Free': '#2ecc71', 'Paid': '#e74c3c'})
    plt.title('Top 5 Installed Categories by Type (Free vs Paid)')
    plt.xlabel('Total Installs (Billions)')
    plt.tight_layout()
    plt.savefig('plots/5_top_categories_by_type.png', dpi=300)
    plt.close()

    print("Top 5 Installed Categories (Free):")
    print(top5_free[['Category', 'Installs']])
    print("\nTop 5 Installed Categories (Paid):")
    print(top5_paid[['Category', 'Installs']])

    # Q2: Category getting reviewed the most
    cat_reviews = df.groupby('Category')['Reviews'].agg(['sum', 'mean', 'count']).sort_values(by='sum', ascending=False).reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=cat_reviews.head(10), x='sum', y='Category', palette='Blues_r')
    plt.title('Top 10 Most Reviewed Categories (Total Reviews)')
    plt.xlabel('Total Reviews (Hundreds of Millions)')
    plt.tight_layout()
    plt.savefig('plots/5_most_reviewed_categories.png', dpi=300)
    plt.close()

    print("\nTop Most Reviewed Category overall:")
    print(cat_reviews.head(1))

    print("\n--- 6. Multivariate Analysis ---")
    # Pairplot across numerical features
    pairplot_cols = ['Rating', 'Reviews_Capped', 'Installs_Capped', 'Size', 'Price_Capped', 'Type']
    g = sns.pairplot(df[pairplot_cols], hue='Type', palette={'Free': '#3498db', 'Paid': '#e74c3c'}, corner=True)
    g.fig.suptitle('Multivariate Pairplot Across Numerical Features', y=1.02)
    plt.savefig('plots/6_pairplot.png', dpi=300)
    plt.close()

    # Heatmap of Category vs Type vs Average Installs
    top10_cats = df['Category'].value_counts().head(10).index
    pivot_df = df[df['Category'].isin(top10_cats)].pivot_table(index='Category', columns='Type', values='Installs', aggfunc='mean')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df / 1e6, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': 'Average Installs (Millions)'})
    plt.title('Heatmap: Top 10 Categories vs Type vs Average Installs (Millions)')
    plt.tight_layout()
    plt.savefig('plots/6_category_type_installs_heatmap.png', dpi=300)
    plt.close()

    print("\n--- 7. Predictive Modeling ---")
    # Prepare Dataset for Regression (Predicting Rating)
    feature_cols = ['Reviews_Capped', 'Size', 'Installs_Capped', 'Price_Capped', 'Update_Recency_Days']
    X = df[feature_cols].copy()
    
    # One-hot encode Category and Type
    X = pd.concat([X, pd.get_dummies(df['Category'], prefix='cat', drop_first=True),
                      pd.get_dummies(df['Type'], prefix='type', drop_first=True)], axis=1)
    y_reg = df['Rating']

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_r, y_train_r)
    y_pred_lr = lr.predict(X_test_r)
    r2_lr = r2_score(y_test_r, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test_r, y_pred_lr))

    # Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_reg.fit(X_train_r, y_train_r)
    y_pred_rf = rf_reg.predict(X_test_r)
    r2_rf = r2_score(y_test_r, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test_r, y_pred_rf))

    print(f"Linear Regression Rating Prediction: R2 = {r2_lr:.4f}, RMSE = {rmse_lr:.4f}")
    print(f"Random Forest Regressor Rating Prediction: R2 = {r2_rf:.4f}, RMSE = {rmse_rf:.4f}")

    # Plot Feature Importance
    importances = rf_reg.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 5))
    feat_importances.plot(kind='barh', color='teal')
    plt.title('Top 10 Feature Importances for Rating Prediction (Random Forest)')
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('plots/7_feature_importance.png', dpi=300)
    plt.close()

    # Classification Model: Predict Rating Category
    y_clf = df['Rating_Category'].astype(str)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_clf.fit(X_train_c, y_train_c)
    y_pred_clf = rf_clf.predict(X_test_c)
    acc_clf = accuracy_score(y_test_c, y_pred_clf)

    print(f"Random Forest Classifier (Rating Category): Accuracy = {acc_clf:.4f}")
    print("Classification Report:")
    print(classification_report(y_test_c, y_pred_clf))

    # Save cleaned data
    df.to_csv('cleaned_googleplaystore.csv', index=False)
    print("\nCleaned dataset saved to 'cleaned_googleplaystore.csv'.")
    print("Analysis complete!")

if __name__ == '__main__':
    main()
