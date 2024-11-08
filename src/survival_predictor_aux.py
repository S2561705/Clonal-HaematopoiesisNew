import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Function to create bins and sample uniformly
def uniform_age_sample(df, n_bins=10, samples_per_bin=None):
    df = df.copy()
    df['age_bin'] = pd.cut(df['age_wave_1'], bins=n_bins)
    
    if samples_per_bin is None:
        samples_per_bin = len(df) // n_bins // 5  # Roughly 20% for test set
    
    test_indices = []
    for bin_name, group in df.groupby('age_bin'):
        if len(group) > samples_per_bin:
            test_indices.extend(group.sample(n=samples_per_bin, random_state=42).index)
        else:
            test_indices.extend(group.index)
    
    train_indices = df.index.difference(test_indices)
    return train_indices, test_indices
# %%
def train_model(data, features, target, train_prop=0.3):
    
    # Create train and test sets
    train_indices, test_indices = uniform_age_sample(data)
    X_train, X_test = data.loc[train_indices, features], data.loc[test_indices, features]
    y_train, y_test = data.loc[train_indices, target], data.loc[test_indices, target]

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define and train models
    models = {
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50, 10), max_iter=1000, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # Train models and store predictions
    predictions = {}
    for name, model in models.items():
        if name == 'MLP':
            model.fit(X_train_scaled, y_train)
            predictions[name] = (model.predict(X_train_scaled), model.predict(X_test_scaled))
        else:
            model.fit(X_train, y_train)
            predictions[name] = (model.predict(X_train), model.predict(X_test))

    # Create ensemble predictions
    ensemble_train = np.mean([pred[0] for pred in predictions.values()], axis=0)
    ensemble_test = np.mean([pred[1] for pred in predictions.values()], axis=0)
    predictions['Ensemble'] = (ensemble_train, ensemble_test)

    # Calculate and print metrics
    for name, (train_pred, test_pred) in predictions.items():
        train_mse = mean_squared_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"{name} Results:")
        print(f"Train MSE: {train_mse:.4f}, R2: {train_r2:.4f}")
        print(f"Test MSE: {test_mse:.4f}, R2: {test_r2:.4f}")
        print()

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, (name, (train_pred, test_pred)) in enumerate(predictions.items()):
        ax = axes[i]
        ax.scatter(y_train, train_pred, color='blue', alpha=0.2, label='Train')
        ax.scatter(y_test, test_pred, color='red', alpha=1, label='Test')
        ax.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'k--', lw=2)
        ax.set_xlabel(f'Actual {target}')
        ax.set_ylabel(f'Predicted {target}')
        ax.set_title(f'{name}: Actual vs Predicted')
        ax.legend()
        ax.text(0.05, 0.95, f'Train R2: {r2_score(y_train, train_pred):.4f}', transform=ax.transAxes)
        ax.text(0.05, 0.90, f'Test R2: {r2_score(y_test, test_pred):.4f}', transform=ax.transAxes)

    plt.tight_layout()
    plt.show()

    # Feature importance plot
    plt.figure()
    x = np.arange(len(features))
    width = 0.35

    for i, (name, model) in enumerate(models.items()):
        if name == 'MLP':
            importance = np.abs(model.coefs_[0]).sum(axis=1)
            importance = importance / np.sum(importance)
        else:
            importance = model.feature_importances_
        
        plt.bar(x + i*width, importance, width, label=name)

    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.xticks(x + width/2, features, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return models
# %%


