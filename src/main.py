# Load the data file
df = pd.read_csv('/content/slope_stability_dataset.csv')  # Read the CSV file

# Show basic info and stats about the data
df.info()
df.describe()

# Show charts for all numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns  # Pick only number columns
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30, color="#4C72B0")
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    sns.despine()
    plt.tight_layout()
    plt.show()

# Show how many times each reinforcement type appears
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='Reinforcement Type', palette="pastel", edgecolor="black")
plt.title('Count Plot for Reinforcement Type')
plt.xlabel('Reinforcement Type')
plt.ylabel('Count')
sns.despine()
plt.tight_layout()
plt.show()

# Boxplot to check spread of Factor of Safety
plt.figure(figsize=(8, 4))
sns.boxplot(y=df['Factor of Safety (FS)'], color="#55A868", width=0.3)
plt.title('Box Plot of Factor of Safety (FS)')
plt.ylabel('Factor of Safety (FS)')
sns.despine()
plt.tight_layout()
plt.show()

# Heatmap to see which numbers are related
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
corr = numeric_df.corr()
sns.heatmap(
    corr,
    annot=True,
    fmt='.2f',
    cmap='crest',
    square=True,
    cbar_kws={'shrink': 0.8},
    linewidths=0.5,
    linecolor='white'
)
plt.title('Correlation Heatmap of Numeric Features')
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()
