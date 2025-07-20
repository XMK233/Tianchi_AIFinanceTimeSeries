#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fund_feature_generator import FundFeatureGenerator
from fund_model_trainer import FundModelTrainer
from datetime import datetime, timedelta

def quick_train():
    print("=== 基金004746快速训练 ===\n")
    
    # 生成特征数据
    print("1. 生成特征数据...")
    generator = FundFeatureGenerator(fund_code="004746")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    features_df = generator.generate_all_features(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        target_days=10
    )
    
    if features_df is None:
        print("❌ 特征生成失败！")
        return None
    
    print(f"✅ 特征生成成功，数据形状: {features_df.shape}")
    
    # 训练模型
    print("\n2. 训练模型...")
    trainer = FundModelTrainer(features_df)
    X, y = trainer.prepare_data()
    X_scaled = trainer.scale_features()
    results = trainer.train_with_rolling_window()
    
    # 评估模型
    print("\n3. 评估模型...")
    eval_df = trainer.evaluate_models()
    
    # 显示最佳模型
    best_model = eval_df.iloc[0]['Model']
    best_r2 = eval_df.iloc[0]['Avg Val R²']
    best_mae = eval_df.iloc[0]['MAE']
    
    print(f"\n🏆 最佳模型: {best_model}")
    print(f"📊 验证集 R²: {best_r2:.4f}")
    print(f"📊 MAE: {best_mae:.4f}")
    
    # 预测最新数据
    print("\n4. 预测未来收益率...")
    latest_data = features_df.iloc[-1:][trainer.feature_cols]
    future_prediction = trainer.predict_future(best_model, latest_data)
    
    print(f"🔮 未来10天收益率预测: {future_prediction:.4f} ({future_prediction*100:.2f}%)")
    
    # 保存模型
    print("\n5. 保存模型...")
    model_filename = trainer.save_model(best_model)
    
    print(f"\n✅ 训练完成！")
    print(f"📁 模型已保存: {model_filename}")
    
    return trainer, eval_df, future_prediction

if __name__ == "__main__":
    trainer, eval_df, prediction = quick_train() 