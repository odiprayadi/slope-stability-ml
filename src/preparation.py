# Separate input (features) and output (target)
target_column = 'Factor of Safety (FS)'
features = df.drop(columns=[target_column])

# Convert the text (Reinforcement Type) into numbers
features = pd.get_dummies(features, columns=['Reinforcement Type'], drop_first=True)

# Set X (inputs) and y (target)
X = features
y = df[target_column]

# Split the data into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
