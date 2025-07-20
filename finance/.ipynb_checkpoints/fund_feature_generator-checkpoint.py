import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FundFeatureGenerator:
    """
    基金特征生成器 - 专门用于生成基金004746的特征
    """
    
    def __init__(self, fund_code="004746"):
        self.fund_code = fund_code
        self.data = None
        
    def get_fund_data(self, start_date=None, end_date=None):
        """
        获取基金历史数据
        """
        try:
            # 获取基金历史净值数据
            fund_data = ak.fund_open_fund_info_em(fund=self.fund_code, indicator="净值")
            
            # 重命名列
            fund_data.columns = ['date', 'nav', 'acc_nav', 'daily_return', 'subscription_status', 'redemption_status', 'dividend']
            
            # 转换日期格式
            fund_data['date'] = pd.to_datetime(fund_data['date'])
            fund_data = fund_data.sort_values('date').reset_index(drop=True)
            
            # 转换数值列
            numeric_cols = ['nav', 'acc_nav', 'daily_return']
            for col in numeric_cols:
                fund_data[col] = pd.to_numeric(fund_data[col], errors='coerce')
            
            # 过滤日期范围
            if start_date:
                fund_data = fund_data[fund_data['date'] >= start_date]
            if end_date:
                fund_data = fund_data[fund_data['date'] <= end_date]
                
            self.data = fund_data
            print(f"成功获取基金{self.fund_code}数据，共{len(fund_data)}条记录")
            return fund_data
            
        except Exception as e:
            print(f"获取基金数据失败: {e}")
            return None
    
    def get_market_data(self, start_date=None, end_date=None):
        """
        获取市场基准数据（沪深300指数）
        """
        try:
            # 获取沪深300指数数据
            market_data = ak.stock_zh_index_daily(symbol="sh000300")
            market_data['date'] = pd.to_datetime(market_data['date'])
            market_data = market_data.sort_values('date').reset_index(drop=True)
            
            # 过滤日期范围
            if start_date:
                market_data = market_data[market_data['date'] >= start_date]
            if end_date:
                market_data = market_data[market_data['date'] <= end_date]
                
            print(f"成功获取沪深300指数数据，共{len(market_data)}条记录")
            return market_data
            
        except Exception as e:
            print(f"获取市场数据失败: {e}")
            return None
    
    def calculate_technical_indicators(self, df, price_col='nav'):
        """
        计算技术指标
        """
        df = df.copy()
        
        # 移动平均线
        df['ma_5'] = df[price_col].rolling(window=5).mean()
        df['ma_10'] = df[price_col].rolling(window=10).mean()
        df['ma_20'] = df[price_col].rolling(window=20).mean()
        df['ma_60'] = df[price_col].rolling(window=60).mean()
        
        # 移动平均线比率
        df['ma_ratio_5_20'] = df['ma_5'] / df['ma_20']
        df['ma_ratio_10_60'] = df['ma_10'] / df['ma_60']
        
        # 价格相对于移动平均线的位置
        df['price_ma5_ratio'] = df[price_col] / df['ma_5']
        df['price_ma20_ratio'] = df[price_col] / df['ma_20']
        
        # 收益率相关指标
        df['daily_return'] = df[price_col].pct_change()
        df['return_5d'] = df[price_col].pct_change(periods=5)
        df['return_10d'] = df[price_col].pct_change(periods=10)
        df['return_20d'] = df[price_col].pct_change(periods=20)
        
        # 波动率指标
        df['volatility_5d'] = df['daily_return'].rolling(window=5).std()
        df['volatility_10d'] = df['daily_return'].rolling(window=10).std()
        df['volatility_20d'] = df['daily_return'].rolling(window=20).std()
        
        # 动量指标
        df['momentum_5d'] = df[price_col] / df[price_col].shift(5) - 1
        df['momentum_10d'] = df[price_col] / df[price_col].shift(10) - 1
        df['momentum_20d'] = df[price_col] / df[price_col].shift(20) - 1
        
        # RSI指标
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['bb_middle'] = df[price_col].rolling(window=20).mean()
        bb_std = df[price_col].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD指标
        exp1 = df[price_col].ewm(span=12).mean()
        exp2 = df[price_col].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 价格通道
        df['highest_20d'] = df[price_col].rolling(window=20).max()
        df['lowest_20d'] = df[price_col].rolling(window=20).min()
        df['channel_position'] = (df[price_col] - df['lowest_20d']) / (df['highest_20d'] - df['lowest_20d'])
        
        return df
    
    def calculate_statistical_features(self, df, price_col='nav'):
        """
        计算统计特征
        """
        df = df.copy()
        
        # 价格分位数
        df['price_percentile_20d'] = df[price_col].rolling(window=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        # 收益率分位数
        df['return_percentile_20d'] = df['daily_return'].rolling(window=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        # 价格偏离度
        df['price_deviation_20d'] = (df[price_col] - df[price_col].rolling(window=20).mean()) / df[price_col].rolling(window=20).std()
        
        # 收益率偏离度
        df['return_deviation_20d'] = (df['daily_return'] - df['daily_return'].rolling(window=20).mean()) / df['daily_return'].rolling(window=20).std()
        
        # 价格变化率
        df['price_change_rate'] = df[price_col].pct_change()
        df['price_change_rate_5d'] = df[price_col].pct_change(periods=5)
        
        # 累计收益率
        df['cumulative_return_5d'] = (1 + df['daily_return']).rolling(window=5).apply(np.prod) - 1
        df['cumulative_return_10d'] = (1 + df['daily_return']).rolling(window=10).apply(np.prod) - 1
        df['cumulative_return_20d'] = (1 + df['daily_return']).rolling(window=20).apply(np.prod) - 1
        
        return df
    
    def calculate_time_features(self, df):
        """
        计算时间特征
        """
        df = df.copy()
        
        # 提取时间特征
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        
        # 是否为月初/月末
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # 是否为季度初/季度末
        df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        
        # 是否为年初/年末
        df['is_year_start'] = df['date'].dt.is_year_start.astype(int)
        df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
        
        # 距离月初/月末的天数
        df['days_from_month_start'] = df['date'].dt.day
        df['days_to_month_end'] = df['date'].dt.days_in_month - df['date'].dt.day
        
        return df
    
    def calculate_market_features(self, fund_df, market_df):
        """
        计算相对于市场的特征
        """
        # 合并基金数据和市场数据
        merged_df = pd.merge(fund_df, market_df[['date', 'close']], on='date', how='left')
        merged_df = merged_df.rename(columns={'close': 'market_close'})
        
        # 计算市场收益率
        merged_df['market_return'] = merged_df['market_close'].pct_change()
        
        # 计算超额收益
        merged_df['excess_return'] = merged_df['daily_return'] - merged_df['market_return']
        
        # 计算相对强度
        merged_df['relative_strength_5d'] = (1 + merged_df['daily_return']).rolling(window=5).apply(np.prod) / \
                                          (1 + merged_df['market_return']).rolling(window=5).apply(np.prod)
        
        merged_df['relative_strength_10d'] = (1 + merged_df['daily_return']).rolling(window=10).apply(np.prod) / \
                                           (1 + merged_df['market_return']).rolling(window=10).apply(np.prod)
        
        merged_df['relative_strength_20d'] = (1 + merged_df['daily_return']).rolling(window=20).apply(np.prod) / \
                                           (1 + merged_df['market_return']).rolling(window=20).apply(np.prod)
        
        # 计算Beta（相对于市场的敏感性）
        merged_df['beta_20d'] = merged_df['daily_return'].rolling(window=20).cov(merged_df['market_return']) / \
                               merged_df['market_return'].rolling(window=20).var()
        
        # 计算Alpha（超额收益的均值）
        merged_df['alpha_20d'] = merged_df['excess_return'].rolling(window=20).mean()
        
        return merged_df
    
    def create_target_variable(self, df, target_days=10):
        """
        创建目标变量：未来10天的累计收益率
        """
        df = df.copy()
        
        # 计算未来N天的累计收益率
        df[f'future_return_{target_days}d'] = df['nav'].shift(-target_days) / df['nav'] - 1
        
        # 创建分类目标变量（可选）
        df[f'future_return_{target_days}d_binary'] = (df[f'future_return_{target_days}d'] > 0).astype(int)
        
        # 创建多分类目标变量（可选）
        def categorize_return(return_val):
            if pd.isna(return_val):
                return np.nan
            elif return_val < -0.05:
                return 0  # 大幅下跌
            elif return_val < 0:
                return 1  # 小幅下跌
            elif return_val < 0.05:
                return 2  # 小幅上涨
            else:
                return 3  # 大幅上涨
        
        df[f'future_return_{target_days}d_category'] = df[f'future_return_{target_days}d'].apply(categorize_return)
        
        return df
    
    def generate_all_features(self, start_date=None, end_date=None, target_days=10):
        """
        生成所有特征
        """
        print("开始生成基金特征...")
        
        # 获取基金数据
        fund_data = self.get_fund_data(start_date, end_date)
        if fund_data is None:
            return None
        
        # 获取市场数据
        market_data = self.get_market_data(start_date, end_date)
        
        # 计算技术指标
        print("计算技术指标...")
        fund_data = self.calculate_technical_indicators(fund_data)
        
        # 计算统计特征
        print("计算统计特征...")
        fund_data = self.calculate_statistical_features(fund_data)
        
        # 计算时间特征
        print("计算时间特征...")
        fund_data = self.calculate_time_features(fund_data)
        
        # 计算市场相关特征
        if market_data is not None:
            print("计算市场相关特征...")
            fund_data = self.calculate_market_features(fund_data, market_data)
        
        # 创建目标变量
        print("创建目标变量...")
        fund_data = self.create_target_variable(fund_data, target_days)
        
        # 删除包含NaN的行
        initial_rows = len(fund_data)
        fund_data = fund_data.dropna()
        final_rows = len(fund_data)
        
        print(f"特征生成完成！")
        print(f"原始数据行数: {initial_rows}")
        print(f"最终数据行数: {final_rows}")
        print(f"删除的NaN行数: {initial_rows - final_rows}")
        
        return fund_data
    
    def get_feature_columns(self, df):
        """
        获取所有特征列名
        """
        # 排除目标变量和基础列
        exclude_cols = ['date', 'nav', 'acc_nav', 'subscription_status', 'redemption_status', 'dividend',
                       'future_return_10d', 'future_return_10d_binary', 'future_return_10d_category']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def save_features(self, df, filename=None):
        """
        保存特征数据
        """
        if filename is None:
            filename = f"fund_{self.fund_code}_features.csv"
        
        df.to_csv(filename, index=False)
        print(f"特征数据已保存到: {filename}")
        
        return filename

def main():
    """
    主函数 - 演示如何使用特征生成器
    """
    # 创建特征生成器
    generator = FundFeatureGenerator(fund_code="004746")
    
    # 设置日期范围（最近2年数据）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    # 生成所有特征
    features_df = generator.generate_all_features(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        target_days=10
    )
    
    if features_df is not None:
        # 显示特征信息
        print("\n=== 特征信息 ===")
        print(f"数据形状: {features_df.shape}")
        print(f"特征数量: {len(generator.get_feature_columns(features_df))}")
        
        # 显示前几行数据
        print("\n=== 数据预览 ===")
        print(features_df.head())
        
        # 显示特征列
        print("\n=== 特征列 ===")
        feature_cols = generator.get_feature_columns(features_df)
        for i, col in enumerate(feature_cols, 1):
            print(f"{i:2d}. {col}")
        
        # 保存特征数据
        generator.save_features(features_df)
        
        # 显示目标变量分布
        print("\n=== 目标变量分布 ===")
        print(f"未来10天收益率统计:")
        print(features_df['future_return_10d'].describe())
        
        print(f"\n未来10天收益率分类分布:")
        print(features_df['future_return_10d_category'].value_counts().sort_index())
        
    else:
        print("特征生成失败！")

if __name__ == "__main__":
    main() 