# import pandas as pd
# import numpy as np
# from datetime import timedelta
# import time
# import xgboost as xgb
# import os, time
# from tqdm import tqdm
# import pyarrow.parquet as pq
# import pyarrow as pa

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# import numpy as np
# import seaborn as sns; sns.set()
# import matplotlib.pyplot as plt

# import sklearn
# import numpy as np
# from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# import tqdm, re, pickle, gc, os, time, random,math,json

# import warnings
# warnings.filterwarnings("ignore")

# from sklearn.model_selection import train_test_split

# from collections import defaultdict

import json
# from pprint import pprint

# pd.set_option('display.max_columns', None)

import deepdiff

def millisec2datetime_1(timestamp):
    time_local = time.localtime(timestamp/1000)
    return time.strftime("%Y-%m-%d %H:%M:%S", time_local)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
        
def get_score_diff_1(df1, df2, epsilon = 1e-8): 
    '''
    这个是看两个评分之间的差异的。
    '''
    def do_something_1(row):
        '''
        获得一个字典，这个字典的key是特征，value是列表。
        列表里面放着的是啥呢？是有差异的trace以及特征值之类的信息。
        '''
        
        feas_traces = defaultdict(list)
        try: 
            for fea in target_feas:
                v1, v2 = row[f"{fea}_x"], row[f"{fea}_y"]
                if (v1 is None) and (v2 is None):## 两者为None，那就是都一样的。
                    continue
                if pd.isna(v1) and pd.isna(v2):## 两者为nan，那就是一样的。
                    continue
                if v1 == v2:## 值一样，那自然也是一样的。
                    continue
                if abs(v1 - v2) <= epsilon:## 差距太小，那自然也是一样的。
                    continue

                diff_record = {
                    "traceid": int(row["trace_id"]),
                    # "loan_account_id": row["loanaccountid"],
                    #"timestamp": row["timestamp"],
                    #"time": millisec2datetime_1(row["timestamp"]),
                    "val_dz": v1, 
                    "val_dq": v2,
                }
                feas_traces[fea].append(diff_record)
        except Exception as e:
            print(e)
        return feas_traces
    
    
    dds = pd.merge(
        df1, df2, on="trace_id", how="inner"
    )
    target_feas = ["score"]
    print(dds.shape)
    
    from pandarallel import pandarallel
    pandarallel.initialize(nb_workers = 4, progress_bar = True)
    df_feas = dds.parallel_apply(lambda row: do_something_1(row), axis = 1)
    feas_traces_dds = defaultdict(list)
    for rst in df_feas:
        for key in rst:
            feas_traces_dds[key].extend(rst[key])
            
    return feas_traces_dds

def get_score_diff_2(df1, df2, fea_list, epsilon = 1e-8, left_symb = "dz", right_symb = "dq", merge_on = "traceid"): 
    '''
    这个是看两份特征之间的差异的。
    '''
    dz, dq = left_symb, right_symb
    def do_something_1(row):
        '''
        获得一个字典，这个字典的key是特征，value是列表。
        列表里面放着的是啥呢？是有差异的trace以及特征值之类的信息。
        '''
        feas_traces = defaultdict(list)
        try: 
            for fea in target_feas:
                v1, v2 = row[f"{fea}_x"], row[f"{fea}_y"]
                
                if (pd.isna(v1) and not pd.isna(v2)) or (not pd.isna(v1) and pd.isna(v2)):
                    ## 一边为空另一边不为空
                    pass
                else:
                    ## 两边都是空或者两个都不是空的情况的判断。有的情况是相等的，有的情况是不等的。
                    if (v1 is None) and (v2 is None):## 两者为None，那就是都一样的。
                        continue
                    if pd.isna(v1) and pd.isna(v2):## 两者为nan，那就是一样的。
                        continue
                    if v1 == v2:## 值一样，那自然也是一样的。
                        continue
                    if abs(v1 - v2) <= epsilon:## 差距太小，那自然也是一样的。
                        continue
                
                diff_record = {
                    "traceid": int(row["traceid"]),
                    "loan_account_id": int(row["loanaccountid_x"]),
                    f"timestamp_{dz}": int(row["timestamp_x"]),
                    f"timestamp_{dq}": int(row["timestamp_y"]),
                    f"time_{dz}": millisec2datetime_1(row["timestamp_x"]),
                    f"time_{dq}": millisec2datetime_1(row["timestamp_y"]),
                    f"val_{dz}": v1, 
                    f"val_{dq}": v2,
                }
                feas_traces[fea].append(diff_record)
        except Exception as e:
            print(e)
            pass
        return feas_traces
    
    dds = pd.merge(
        df1, df2, on=merge_on, how="inner"
    )
    target_feas = fea_list
    print(dds.shape)
    # print(target_feas)
    
    from pandarallel import pandarallel
    pandarallel.initialize(nb_workers = 4, progress_bar = True)
    df_feas = dds.parallel_apply(lambda row: do_something_1(row), axis = 1)
    feas_traces_dds = defaultdict(list)
    for rst in df_feas:
        for key in rst:
            feas_traces_dds[key].extend(rst[key])
    ## 把所有的diff的情况给它打出来。        
    with open("feas_diffVals.txt", "w") as f:
        json.dump(feas_traces_dds, f, cls=NpEncoder) # indent=4,
    ## 
    traceid_feaList = defaultdict(dict)
    for fea in feas_traces_dds:
        l = feas_traces_dds[fea]
        for item in l:
            tid = str(item["traceid"])

            traceid_feaList[tid]["traceid"] = item["traceid"]
            traceid_feaList[tid]["loan_account_id"] = item["loan_account_id"]
            traceid_feaList[tid][f"timestamp_{dz}"] = item[f"timestamp_{dz}"]
            traceid_feaList[tid][f"timestamp_{dq}"] = item[f"timestamp_{dq}"]
            traceid_feaList[tid][f"time_{dz}"] = item[f"time_{dz}"]
            traceid_feaList[tid][f"time_{dq}"] = item[f"time_{dq}"]

            if "feas" not in traceid_feaList[tid]:
                traceid_feaList[tid]["feas"] = {}
            traceid_feaList[tid]["feas"][fea] = {
                f"val_{dz}": item[f"val_{dz}"], 
                f"val_{dq}": item[f"val_{dq}"]
            }
    with open("traceid_feaVal.txt", "w") as f:
        json.dump(traceid_feaList, f, cls=NpEncoder) # indent=4,        
    ## 按照等级对特征进行归类。
    s_l = sorted(list(feas_traces_dds.keys()))
    dic = {}
    for i in s_l:
        parts = i.split("___")
        if parts[0] not in dic:
            dic[parts[0]] = {}
        cur_dic = dic[parts[0]]
        for part in parts[1:-1]:
            if part not in cur_dic:
                cur_dic[part] = {}
            cur_dic = cur_dic[part]
        cur_dic[parts[-1]] = i
    with open("cascaded_fea_names.txt", "w") as f:
        json.dump(dic, f)

    return feas_traces_dds
