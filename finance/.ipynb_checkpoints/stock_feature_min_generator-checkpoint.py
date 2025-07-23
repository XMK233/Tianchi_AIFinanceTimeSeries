import akshare as ak
import pandas as pd
import numpy as np

class StockFeatureMinGenerator:
    def __init__(self, stock_code, period="1"):
        self.stock_code = stock_code
        self.period = period  # "1" for 1min, "5" for 5min, etc.
        self.data = None

    def get_stock_min_data(self):
        df = ak.stock_zh_a_minute(symbol=self.stock_code, period=self.period, adjust="qfq")
        df = df.rename(columns={
            '时间': 'datetime', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
            '成交量': 'volume', '成交额': 'amount'
        })
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        for col in ['open', 'close', 'high', 'low', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        self.data = df
        print(f"获取股票{self.stock_code}分钟线数据，共{len(df)}条记录")
        return df

    def calculate_technical_indicators(self, df):
        df = df.copy()
        for w in [5, 10, 20, 30, 60, 120]:
            df[f'ma_{w}'] = df['close'].rolling(window=w).mean()
        df['volatility_10'] = df['close'].pct_change().rolling(window=10).std()
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        return df

    def calculate_statistical_features(self, df):
        df = df.copy()
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)
        df['return_20'] = df['close'].pct_change(20)
        df['max_up_20'] = df['return_1'].rolling(window=20).max()
        df['max_down_20'] = df['return_1'].rolling(window=20).min()
        return df

    def calculate_time_features(self, df):
        df = df.copy()
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['dayofweek'] = df['datetime'].dt.dayofweek
        return df

    def create_target_variable(self, df, target_minutes=10):
        df = df.copy()
        df[f'future_return_{target_minutes}min'] = df['close'].shift(-target_minutes) / df['close'] - 1
        df[f'future_return_{target_minutes}min_binary'] = (df[f'future_return_{target_minutes}min'] > 0).astype(int)
        return df

    def generate_all_features(self):
        print("生成股票分钟线特征...")
        df = self.get_stock_min_data()
        df = self.calculate_technical_indicators(df)
        df = self.calculate_statistical_features(df)
        df = self.calculate_time_features(df)
        for mins in [1, 5, 10, 30]:
            df = self.create_target_variable(df, target_minutes=mins)
        df = df.dropna().reset_index(drop=True)
        print(f"特征生成完成，数据形状: {df.shape}")
        return df

    def get_feature_columns(self, df):
        exclude_cols = ['datetime', 'close', 'open', 'high', 'low', 'volume', 'amount']
        for m in [1, 5, 10, 30]:
            exclude_cols += [f'future_return_{m}min', f'future_return_{m}min_binary']
        return [col for col in df.columns if col not in exclude_cols]

    def save_features(self, df, filename=None):
        if filename is None:
            filename = f"stock_{self.stock_code}_min_features.csv"
        df.to_csv(filename, index=False)
        print(f"特征数据已保存到: {filename}")
        return filename

def main():
    stock_code = "600519"  # 茅台示例
    generator = StockFeatureMinGenerator(stock_code, period="1")
    features_df = generator.generate_all_features()
    print(features_df.head())
    generator.save_features(features_df)

if __name__ == "__main__":
    main() 