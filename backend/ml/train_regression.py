import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pickle


def prepare_features(df):

    df_processed = df.copy()
    
    df_processed[['base_currency', 'quote_currency']] = df_processed['slug'].str.split('/', expand=True)
    
    base_currency_dummies = pd.get_dummies(df_processed['base_currency'], prefix='base')
    quote_currency_dummies = pd.get_dummies(df_processed['quote_currency'], prefix='quote')
    
    df_processed['date'] = pd.to_datetime(df_processed['date'])
    df_processed['year'] = df_processed['date'].dt.year
    df_processed['month'] = df_processed['date'].dt.month
    df_processed['day'] = df_processed['date'].dt.day
    df_processed['day_of_week'] = df_processed['date'].dt.dayofweek
    df_processed['quarter'] = df_processed['date'].dt.quarter
    
    df_processed['price_range'] = df_processed['high'] - df_processed['low']
    df_processed['price_change'] = df_processed['close'] - df_processed['open']
    df_processed['daily_return'] = (df_processed['close'] - df_processed['open']) / df_processed['open']
    df_processed['high_low_ratio'] = df_processed['high'] / df_processed['low']
    df_processed['close_open_ratio'] = df_processed['close'] / df_processed['open']
    
    currency_dummies = pd.get_dummies(df_processed['currency'], prefix='currency')
    
    numeric_features = ['open', 'high', 'low', 'price_range', 'price_change', 'daily_return', 'high_low_ratio', 'close_open_ratio',
                        'year', 'month', 'day', 'day_of_week', 'quarter']
    
    X = df_processed[numeric_features].copy()
    X = pd.concat([X, base_currency_dummies, quote_currency_dummies, currency_dummies], axis=1)

    y = df_processed['close']
    
    return X, y, df_processed

def train_model(filename: str, model_name: str, train_size: float = 0.8, model_type: str = 'linear'):

    path = os.path.join('data', filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл {path} не найден")
    
    df = pd.read_csv(path)
    
    X, y, df_processed = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=train_size, 
        random_state=42,
        shuffle=True
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == 'linear':
        model = LinearRegression()
        model_params = {}
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
        model_params = {'alpha': 1.0}
    elif model_type == 'decision_tree':
        model = DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model_params = {
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    model.fit(X_train_scaled, y_train)
    
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join('models', f"{model_name}.pkl")
    scaler_path = os.path.join('models', f"{model_name}_scaler.pkl")
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': X.columns.tolist(),
        'model_type': model_type,
        'model_params': model_params,
        'train_size': train_size
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    results = {
        'model_type': model_type,
        'model_path': model_path,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': X.shape[1],
        'features': X.columns.tolist()[:10]
    }
    
    return results

def train_all_models(filename: str, train_size: float = 0.8):
    results = {}
    if model_type:
        model_types = [model_type]
    else:
      model_types = ['linear', 'ridge', 'decision_tree']
    
    for model_type in model_types:
        model_name = f"{filename.split('.')[0]}_{model_type}"
        try:
            result = train_model(filename, model_name, train_size, model_type)
            results[model_type] = result
            print(f"{model_type}: RMSE = {result['test_rmse']:.4f}, R² = {result['test_r2']:.4f}")
        except Exception as e:
            print(f"Ошибка при обучении {model_type}: {str(e)}")
    
    return results   