import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

class FundModelClassificationTrainer:
    def __init__(self, features_df, target_days=10):
        self.features_df = features_df
        self.target_days = target_days
        self.target_col = f'future_return_{target_days}d_binary'
        self.feature_cols = None
        self.scaler = None
        self.models = {}
        self.results = {}

    def prepare_data(self):
        exclude_cols = ['date', 'nav', 'daily_return']
        for d in [1, 5, 10, 30]:
            exclude_cols += [f'future_return_{d}d', f'future_return_{d}d_binary', f'future_return_{d}d_category']
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
        print("使用递增窗口训练分类模型...")
        models = {
            'Logistic Regression': LogisticRegression(max_iter=200),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        }
        for name in models.keys():
            self.results[name] = {
                'val_scores': [], 'predictions': [], 'probas': [], 'actuals': []
            }
        n = len(self.X_scaled)
        for i in tqdm(range(1, n), desc='Expanding Window'):
            X_train = self.X_scaled[:i]
            y_train = self.y.iloc[:i]
            X_test = self.X_scaled[i:i+1]
            y_test = self.y.iloc[i:i+1]
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                self.results[name]['val_scores'].append(accuracy_score(y_test, y_pred))
                self.results[name]['predictions'].extend(y_pred)
                self.results[name]['probas'].extend(y_proba)
                self.results[name]['actuals'].extend(y_test.values)
                self.models[name] = model
        for name in models.keys():
            acc = accuracy_score(self.results[name]['actuals'], self.results[name]['predictions'])
            f1 = f1_score(self.results[name]['actuals'], self.results[name]['predictions'])
            auc = roc_auc_score(self.results[name]['actuals'], self.results[name]['probas'])
            self.results[name]['acc'] = acc
            self.results[name]['f1'] = f1
            self.results[name]['auc'] = auc
        return self.results

    def evaluate_models(self):
        print("\n=== 分类模型评估结果 ===")
        eval_results = []
        for name, results in self.results.items():
            eval_results.append({
                'Model': name,
                'ACC': results['acc'],
                'F1': results['f1'],
                'AUC': results['auc']
            })
        eval_df = pd.DataFrame(eval_results)
        eval_df = eval_df.sort_values('AUC', ascending=False)
        print(eval_df.round(4))
        return eval_df

    def plot_predictions(self, model_name):
        if model_name not in self.results:
            print(f"模型 {model_name} 不存在")
            return
        results = self.results[model_name]
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(results['actuals'], label='实际', alpha=0.7)
        plt.plot(results['predictions'], label='预测', alpha=0.7)
        plt.xlabel('时间')
        plt.ylabel('类别')
        plt.title(f'{model_name} - 分类结果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.plot(results['probas'], label='预测概率', alpha=0.7)
        plt.xlabel('时间')
        plt.ylabel('正类概率')
        plt.title(f'{model_name} - 正类概率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{model_name}_classification_{self.target_days}d.png', dpi=300, bbox_inches='tight')
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
        proba = model.predict_proba(latest_features_scaled)[:, 1] if hasattr(model, 'predict_proba') else model.predict(latest_features_scaled)
        return proba[0]

    def save_model(self, model_name):
        import pickle
        filename = f"{model_name.replace(' ', '_').lower()}_classification_model_{self.target_days}d.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        print(f"模型已保存: {filename}")
        return filename

def main():
    from fund_feature_generator import FundFeatureGenerator
    generator = FundFeatureGenerator(fund_code="004746")
    features_df = generator.generate_all_features(target_days=10)
    trainer = FundModelClassificationTrainer(features_df, target_days=10)
    X, y = trainer.prepare_data()
    X_scaled = trainer.scale_features()
    results = trainer.train_with_expanding_window()
    eval_df = trainer.evaluate_models()
    best_model = eval_df.iloc[0]['Model']
    trainer.plot_predictions(best_model)
    latest_data = features_df.iloc[-1:][trainer.feature_cols]
    future_proba = trainer.predict_future(best_model, latest_data)
    print(f"未来{trainer.target_days}天为正的概率预测: {future_proba:.4f}")
    trainer.save_model(best_model)
    return trainer, eval_df

if __name__ == "__main__":
    trainer, eval_df = main() 