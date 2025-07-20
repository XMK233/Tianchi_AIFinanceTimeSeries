# 基金004746收益率预测模型

使用机器学习预测基金004746未来10天的收益率。

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 快速训练
```bash
python quick_train_example.py
```

### 完整训练
```bash
python fund_model_trainer.py
```

## 文件说明

- `fund_feature_generator.py` - 特征生成器
- `fund_model_trainer.py` - 模型训练器（滚动窗口）
- `quick_train_example.py` - 快速训练示例
- `test_code.py` - 代码测试

## 功能特点

### 特征工程
- 技术指标：MA、RSI、MACD、布林带、动量指标
- 统计特征：分位数、偏离度、累计收益率
- 时间特征：年、月、日、特殊时点
- 市场特征：超额收益、相对强度、Beta、Alpha

### 机器学习模型
- 线性模型：线性回归、岭回归
- 集成模型：随机森林、LightGBM、XGBoost
- 评估指标：MAE、RMSE、R²

### 滚动窗口训练
- 使用252天（约1年）作为训练窗口
- 每7天（约1周）滑动一次
- 避免时间序列数据泄露

## 使用示例

```python
from fund_feature_generator import FundFeatureGenerator
from fund_model_trainer import FundModelTrainer
from datetime import datetime, timedelta

# 生成特征
generator = FundFeatureGenerator(fund_code="004746")
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

features_df = generator.generate_all_features(
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d'),
    target_days=10
)

# 训练模型
trainer = FundModelTrainer(features_df)
X, y = trainer.prepare_data()
X_scaled = trainer.scale_features()
results = trainer.train_with_rolling_window()

# 评估模型
eval_df = trainer.evaluate_models()
print(eval_df)

# 预测未来收益率
latest_data = features_df.iloc[-1:][trainer.feature_cols]
prediction = trainer.predict_future("LightGBM", latest_data)
print(f"未来10天收益率预测: {prediction:.4f}")
```

## 输出结果

运行后生成：
- 特征数据文件 (`fund_004746_features.csv`)
- 训练好的模型文件 (`*_model.pkl`)
- 预测结果图 (`*_predictions.png`)

## 注意事项

1. 需要网络连接获取akshare数据
2. 基金数据可能有缺失，代码会自动处理
3. 金融预测具有不确定性，模型仅供参考
4. 滚动窗口训练时间较长，请耐心等待
