import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class StockFeatureGenerator:
    def __init__(self, stock_code):
        self.stock_code = stock_code
        self.data = None

    def get_stock_data(self):
        # 获取A股日线行情（前复权）
        df = ak.stock_zh_a_hist(symbol=self.stock_code, period="daily", adjust="qfq")
        df = df.rename(columns={
            '日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
            '成交量': 'volume', '成交额': 'amount', '涨跌幅': 'pct_chg', '换手率': 'turnover_rate'
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        for col in ['open', 'close', 'high', 'low', 'volume', 'amount', 'pct_chg', 'turnover_rate']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        self.data = df
        print(f"获取股票{self.stock_code}数据，共{len(df)}条记录")
        return df

    def calculate_technical_indicators(self, df):
        df = df.copy()
        # 均线
        for w in [5, 10, 20, 30, 60, 120, 250]:
            df[f'ma_{w}'] = df['close'].rolling(window=w).mean()
        # 波动率
        df['volatility_10d'] = df['close'].pct_change().rolling(window=10).std()
        df['volatility_20d'] = df['close'].pct_change().rolling(window=20).std()
        # 动量
        df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        # 布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df

    def calculate_statistical_features(self, df):
        df = df.copy()
        # 收益率分布
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)
        # 过去20天最大/最小涨跌幅
        df['max_up_20d'] = df['return_1d'].rolling(window=20).max()
        df['max_down_20d'] = df['return_1d'].rolling(window=20).min()
        # 价格分位数
        df['close_percentile_20d'] = df['close'].rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        return df

    def calculate_time_features(self, df):
        df = df.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        return df

    def create_target_variable(self, df, target_days=10):
        df = df.copy()
        df[f'future_return_{target_days}d'] = df['close'].shift(-target_days) / df['close'] - 1
        df[f'future_return_{target_days}d_binary'] = (df[f'future_return_{target_days}d'] > 0).astype(int)
        return df

    def generate_all_features(self):
        print("生成股票特征...")
        df = self.get_stock_data()
        df = self.calculate_technical_indicators(df)
        df = self.calculate_statistical_features(df)
        df = self.calculate_time_features(df)
        for days in [1, 5, 10, 30]:
            df = self.create_target_variable(df, target_days=days)
        df = df.dropna().reset_index(drop=True)
        print(f"特征生成完成，数据形状: {df.shape}")
        return df

    def get_feature_columns(self, df):
        exclude_cols = ['date', 'close', 'open', 'high', 'low', 'volume', 'amount', 'pct_chg', 'turnover_rate']
        for d in [1, 5, 10, 30]:
            exclude_cols += [f'future_return_{d}d', f'future_return_{d}d_binary']
        return [col for col in df.columns if col not in exclude_cols]

    def save_features(self, df, filename=None):
        if filename is None:
            filename = f"stock_{self.stock_code}_features.csv"
        df.to_csv(filename, index=False)
        print(f"特征数据已保存到: {filename}")
        return filename

def main():
    stock_code = "600519"  # 茅台示例
    generator = StockFeatureGenerator(stock_code)
    features_df = generator.generate_all_features()
    print(features_df.head())
    generator.save_features(features_df)

if __name__ == "__main__":
    main() 