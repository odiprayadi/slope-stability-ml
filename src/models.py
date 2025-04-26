# Define models we want to try
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(),
    'MLP': MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500)
}

# Train models and check how good they are
results = []
for name, model in models.items():
    model.fit(X_train, y_train)  # Train
    y_pred = model.predict(X_test)  # Predict
    r2 = r2_score(y_test, y_pred)  # Accuracy
    mae = mean_absolute_error(y_test, y_pred)  # Error
    results.append((name, r2, mae))

# Save and show the results in a table
results_df = pd.DataFrame(results, columns=['Model', 'R² Score', 'MAE'])

# Plot model accuracy (R²)
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='R² Score', y='Model', palette='Blues_r')
plt.title('Model Comparison by R² Score')
plt.xlabel('R² Score')
plt.ylabel('Model')
sns.despine()
plt.tight_layout()
plt.show()

# Plot model error (MAE)
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='MAE', y='Model', palette='Blues_r')
plt.title('Model Comparison by MAE')
plt.xlabel('Mean Absolute Error')
plt.ylabel('Model')
sns.despine()
plt.tight_layout()
plt.show()
