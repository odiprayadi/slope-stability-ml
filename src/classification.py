# Function to give label: stable, caution, or unstable
def fs_label(fs):
    if fs >= 1.5:
        return 'Stable'
    elif fs >= 1.2:
        return 'Caution'
    else:
        return 'Unstable'

# Add new column with FS class
df['FS_Class'] = df['Factor of Safety (FS)'].apply(fs_label)

# Set X and y for classification
X_cls = features
y_cls = df['FS_Class']
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Train classification model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_cls, y_train_cls)
y_pred_cls = clf.predict(X_test_cls)

# Show how well the classifier did
print(classification_report(y_test_cls, y_pred_cls))

# Show confusion matrix as a heatmap
conf_matrix = confusion_matrix(y_test_cls, y_pred_cls)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
sns.despine()
plt.tight_layout()
plt.show()

# Check which features are most important
perm_importance = permutation_importance(clf, X_test_cls, y_test_cls, n_repeats=10, random_state=42)
importances_df = pd.DataFrame({
    'feature': X_test_cls.columns,
    'importance': perm_importance.importances_mean
}).sort_values(by='importance', ascending=True)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=importances_df, x='importance', y='feature', palette='Blues_r')
plt.title('Permutation Importance of Features')
plt.xlabel('Mean Permutation Importance')
plt.ylabel('Feature')
sns.despine()
plt.tight_layout()
plt.show()
