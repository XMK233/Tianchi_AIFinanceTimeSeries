import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import tqdm

class StockModelMinTrainer:
    def __init__(self, features_df, target_minutes=10):
        self.features_df = features_df
        self.target_minutes = target_minutes
        self.target_col = f'future_return_{target_minutes}min'
        self.feature_cols = None
        self.scaler = None
        self.models = {}
        self.results = {}

    def prepare_data(self):
        exclude_cols = ['datetime', 'close', 'open', 'high', 'low', 'volume', 'amount']
        for m in [1, 5, 10, 30]:
            exclude_cols += [f'future_return_{m}min', f'future_return_{m}min_binary']
        self.feature_cols = [col for col in self.features_df.columns if col not in exclude_cols]
        self.X = self.features_df[self.feature_cols].copy()
        self.y = self.features_df[self.target_col].copy()
        self.X = self.X.replace([np.inf, -np.inf], np.nan)
        num_cols = self.X.select_dtypes(include=[np.number]).columns
        self.X[num_cols] = self.X[num_cols].fillna(self.X[num_cols].median())
        print(f"数据形状: {self.X.shape}")
        return self.X, self.y

    def scale_features(self, scaler_type='standard'):
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        return self.X_scaled

    def train_with_sparse_expanding_window(self, step=100):
        print("使用稀疏递增窗口训练模型...")
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
        for i in tqdm.tqdm(range(100, n, step)):
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
        plt.savefig(f'{model_name}_min_predictions_{self.target_minutes}min.png', dpi=300, bbox_inches='tight')
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
        prediction = model.predict(latest_features_scaled)
        return prediction[0] if hasattr(prediction, '__len__') else prediction

    def save_model(self, model_name):
        import pickle
        filename = f"{model_name.replace(' ', '_').lower()}_min_model_{self.target_minutes}min.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        print(f"模型已保存: {filename}")
        return filename

def main():
    from stock_feature_min_generator import StockFeatureMinGenerator
    stock_code = "600519"  # 茅台示例
    generator = StockFeatureMinGenerator(stock_code, period="1")
    features_df = generator.generate_all_features()
    trainer = StockModelMinTrainer(features_df, target_minutes=10)
    X, y = trainer.prepare_data()
    X_scaled = trainer.scale_features()
    results = trainer.train_with_sparse_expanding_window(step=100)
    eval_df = trainer.evaluate_models()
    best_model = eval_df.iloc[0]['Model']
    trainer.plot_predictions(best_model)
    latest_data = features_df.iloc[-1:][trainer.feature_cols]
    future_prediction = trainer.predict_future(best_model, latest_data)
    print(f"未来{trainer.target_minutes}分钟收益率预测: {future_prediction:.4f} ({future_prediction*100:.2f}%)")
    trainer.save_model(best_model)
    return trainer, eval_df

if __name__ == "__main__":
    trainer, eval_df = main() 