import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')
import tqdm

class DNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
        super(DNNRegressor, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class StockModelTrainer:
    def __init__(self, features_df, target_days=10):
        self.features_df = features_df
        self.target_days = target_days
        self.target_col = f'future_return_{target_days}d'
        self.feature_cols = None
        self.scaler = None
        self.models = {}
        self.results = {}

    def prepare_data(self):
        exclude_cols = ['date', 'close', 'open', 'high', 'low', 'volume', 'amount', 'pct_chg', 'turnover_rate']
        for d in [1, 5, 10, 30]:
            exclude_cols += [f'future_return_{d}d', f'future_return_{d}d_binary']
        self.feature_cols = [col for col in self.features_df.columns if col not in exclude_cols]
        self.X = self.features_df[self.feature_cols].copy()
        self.y = self.features_df[self.target_col].copy()
        self.X = self.X.replace([np.inf, -np.inf], np.nan)
        self.X = self.X.fillna(self.X.median())
        print(f"数据形状: {self.X.shape}")
        return self.X, self.y

    def scale_features(self, scaler_type='standard'):
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        return self.X_scaled

    def train_with_expanding_window(self):
        print("使用递增窗口训练模型...")
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        for name in models.keys():
            self.results[name] = {
                'val_scores': [], 'predictions': [], 'actuals': []
            }
        n = len(self.X_scaled)
        for i in tqdm.tqdm(range(1, n)):
            if i < 100:
                continue  # 训练集小于100时跳过
            X_train = self.X_scaled[:i]
            y_train = self.y.iloc[:i]
            X_test = self.X_scaled[i:i+1]
            y_test = self.y.iloc[i:i+1]
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                val_score = r2_score(y_test, y_pred)
                self.results[name]['val_scores'].append(val_score)
                self.results[name]['predictions'].extend(y_pred)
                self.results[name]['actuals'].extend(y_test.values)
                self.models[name] = model
        for name in models.keys():
            avg_val_score = np.mean(self.results[name]['val_scores'])
            mae = mean_absolute_error(self.results[name]['actuals'], self.results[name]['predictions'])
            rmse = np.sqrt(mean_squared_error(self.results[name]['actuals'], self.results[name]['predictions']))
            self.results[name]['avg_val_score'] = avg_val_score
            self.results[name]['mae'] = mae
            self.results[name]['rmse'] = rmse
        return self.results

    def train_dnn(self, epochs=50, batch_size=32, lr=0.001):
        print("训练DNN模型...")
        dnn_model = DNNRegressor(input_size=len(self.feature_cols))
        split_idx = int(len(self.X_scaled) * 0.8)
        X_train = self.X_scaled[:split_idx]
        y_train = self.y.iloc[:split_idx]
        X_val = self.X_scaled[split_idx:]
        y_val = self.y.iloc[split_idx:]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dnn_model = dnn_model.to(device)
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val.values).to(device)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(dnn_model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        dnn_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = dnn_model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")
        dnn_model.eval()
        with torch.no_grad():
            train_pred = dnn_model(X_train_tensor).squeeze().cpu().numpy()
            val_pred = dnn_model(X_val_tensor).squeeze().cpu().numpy()
        train_score = r2_score(y_train, train_pred)
        val_score = r2_score(y_val, val_pred)
        mae = mean_absolute_error(y_val, val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        self.results['DNN'] = {
            'train_scores': [train_score],
            'val_scores': [val_score],
            'predictions': val_pred.tolist(),
            'actuals': y_val.values.tolist(),
            'avg_train_score': train_score,
            'avg_val_score': val_score,
            'mae': mae,
            'rmse': rmse
        }
        self.models['DNN'] = dnn_model
        print(f"DNN训练完成 - 验证集R²: {val_score:.4f}, MAE: {mae:.4f}")
        return dnn_model

    def evaluate_models(self):
        print("\n=== 模型评估结果 ===")
        eval_results = []
        for name, results in self.results.items():
            eval_results.append({
                'Model': name,
                'Avg Val R²': results['avg_val_score'],
                'MAE': results['mae'],
                'RMSE': results['rmse']
            })
        eval_df = pd.DataFrame(eval_results)
        eval_df = eval_df.sort_values('Avg Val R²', ascending=False)
        print(eval_df.round(4))
        return eval_df

    def plot_predictions(self, model_name):
        if model_name not in self.results:
            print(f"模型 {model_name} 不存在")
            return
        results = self.results[model_name]
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(results['actuals'], results['predictions'], alpha=0.6)
        plt.plot([min(results['actuals']), max(results['actuals'])],
                [min(results['actuals']), max(results['actuals'])], 'r--', lw=2)
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title(f'{model_name} - 预测 vs 实际值')
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.plot(results['actuals'], label='实际值', alpha=0.7)
        plt.plot(results['predictions'], label='预测值', alpha=0.7)
        plt.xlabel('时间')
        plt.ylabel('收益率')
        plt.title(f'{model_name} - 时间序列预测结果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{model_name}_predictions_{self.target_days}d.png', dpi=300, bbox_inches='tight')
        plt.show()

    def predict_future(self, model_name, latest_features):
        if model_name not in self.models:
            print(f"模型 {model_name} 不存在")
            return None
        model = self.models[model_name]
        latest_features = latest_features[self.feature_cols]
        if hasattr(self, 'scaler'):
            latest_features_scaled = self.scaler.transform(latest_features)
        else:
            latest_features_scaled = latest_features
        if model_name == 'DNN':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(latest_features_scaled).to(device)
                prediction = model(X_tensor).squeeze().cpu().numpy()
        else:
            prediction = model.predict(latest_features_scaled)
        return prediction[0] if hasattr(prediction, '__len__') else prediction

    def save_model(self, model_name):
        import pickle
        filename = f"{model_name.replace(' ', '_').lower()}_model_{self.target_days}d.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        print(f"模型已保存: {filename}")
        return filename

def main():
    from stock_feature_generator import StockFeatureGenerator
    stock_code = "600519"  # 茅台示例
    generator = StockFeatureGenerator(stock_code)
    features_df = generator.generate_all_features()
    trainer = StockModelTrainer(features_df, target_days=10)
    X, y = trainer.prepare_data()
    X_scaled = trainer.scale_features()
    results = trainer.train_with_expanding_window()
    eval_df = trainer.evaluate_models()
    best_model = eval_df.iloc[0]['Model']
    trainer.plot_predictions(best_model)
    latest_data = features_df.iloc[-1:][trainer.feature_cols]
    future_prediction = trainer.predict_future(best_model, latest_data)
    print(f"未来{trainer.target_days}天收益率预测: {future_prediction:.4f} ({future_prediction*100:.2f}%)")
    trainer.save_model(best_model)
    return trainer, eval_df

if __name__ == "__main__":
    trainer, eval_df = main() 