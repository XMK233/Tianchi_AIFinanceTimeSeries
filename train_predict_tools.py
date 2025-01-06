# import pandas as pd
# import numpy as np
# from datetime import timedelta, datetime
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

# import tqdm, re, pickle, gc, os, time, random, sys

# import warnings
# warnings.filterwarnings("ignore")

# from sklearn.model_selection import train_test_split

# from collections import defaultdict

# import json
# from pprint import pprint

# pd.set_option('display.max_columns', None)

# import deepdiff, glob
# import math

# from sklearn.model_selection import GridSearchCV,ParameterGrid,StratifiedKFold,train_test_split,GroupKFold
# from sklearn import preprocessing
# from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, ElasticNet, ElasticNetCV, LogisticRegressionCV

# from PIL import Image
# from PIL import ImageDraw
# from PIL import ImageFont
# from IPython.display import display
# from matplotlib.pyplot import imshow


def train_model_with_different_label_2(dtrain, dtest, saved_model_name, do_valid = True, save_model_path = "/data/private/public_data/cpu1/minkexiu/DuoTouSubModels/trained_models"):
    
    params={
            'booster':'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 7, #5,
            'scale_pos_weight': 3,
            'learning_rate': 0.05, # 0.1, 
            'reg_lambda': 7, #5,
            'reg_alpha': 100, # 0, 
            'colsample_bytree': 0.8,
            'tree_method': 'gpu_hist',
            "gpu_id": 0,
#             "n_gpus": -1,
        }
    params['nthread'] = 25
    train_start_time = time.time()
    if do_valid:
        booster_maidian = xgb.train(
            params, dtrain, num_boost_round=1000000, evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=200,verbose_eval=100
        )
    else:
        booster_maidian = xgb.train(
            params, dtrain, num_boost_round=10000, 
            # evals=[(dtrain, 'train'), (dtest, 'test')],
            # early_stopping_rounds=50,
            verbose_eval=100
        )
    train_end_time = time.time()
    
    print(f"train time: {train_end_time - train_start_time}")
    
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    booster_maidian.save_model(os.path.join(save_model_path, saved_model_name))
    
    return booster_maidian

def train_model_with_different_label_3(
    dtrain, dtest, params, 
    saved_model_name, 
    save_model_path = "/data/private/public_data/cpu1/minkexiu/DuoTouSubModels/trained_models",
    save_model = True,
):
    params['nthread'] = 25
    train_start_time = time.time()
    booster_maidian = xgb.train(params, dtrain, num_boost_round=1000000, evals=[(dtrain, 'train'), (dtest, 'test')],
                                                      early_stopping_rounds=200,
                                verbose_eval=500
                               )
    train_end_time = time.time()
    
    print(f"train time: {train_end_time - train_start_time}")
    
#     if not os.path.exists("trained_models"):
#         os.makedirs("trained_models")
        
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    if save_model:
        booster_maidian.save_model(os.path.join(save_model_path, saved_model_name))
        
    return booster_maidian

def train_model_with_different_label_2_variousParam_nbr(
    dtrain, dtest, saved_model_name, 
    do_valid = True, save_model_path = "/data/private/public_data/cpu1/minkexiu/DuoTouSubModels/trained_models", params = None,
    nbr = 10000 ## 这个就可以用来设置模型的树的数量
):
    
    if params is None:
        params={
                'booster':'gbtree',
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 7, #5,
                'scale_pos_weight': 3,
                'learning_rate': 0.05, # 0.1, 
                'reg_lambda': 7, #5,
                'reg_alpha': 100, # 0, 
                'colsample_bytree': 0.8,
                'tree_method': 'gpu_hist',
    #             "gpu_id": 5,
                "n_gpus": -1,
            }
    print(params)
    params['nthread'] = 25
    train_start_time = time.time()
    if do_valid:
        booster_maidian = xgb.train(
            params, dtrain, num_boost_round=nbr, evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=200,verbose_eval=100
        )
    else:
        booster_maidian = xgb.train(
            params, dtrain, num_boost_round=nbr, 
            evals=[(dtrain, 'train')], # , (dtest, 'test')
            # early_stopping_rounds=50,
            verbose_eval=100
        )
    train_end_time = time.time()
    
    print(f"train time: {train_end_time - train_start_time}")
    
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    booster_maidian.save_model(os.path.join(save_model_path, saved_model_name))
    
    return booster_maidian

def predict_with_trained_model_v3(model_name, feas_to_be_used, oot_append, 
                                  predict_data_batch = 5,
                                 model_dir = f"/data/private/public_data/cpu1/minkexiu/DuoTouSubModels/trained_models", 
                                  actual_model = None
                                 ): 
    '''
    model_name: str of the model's name. 
    oot_append: dataframe.
    predict_data_batch: how many batches the oot_append will be splitted. 
    actual_model: directly pass a model here, not changing 
    '''
    model = xgb.Booster({'nthread': 4}) ## ns: not strict labeled
    model.load_model(
        os.path.join(model_dir, model_name)
    )
    
    feas_cols_v2 = feas_to_be_used
    
    print(f"How many features are used? ", len(feas_cols_v2))
    
    n_batches = predict_data_batch
    nrow_interval = oot_append.shape[0] // n_batches - 5 ## 把原来的数据分成5份，然后逐份来做预测。否则实在太久。

    ns_prob_col = None

    for i in tqdm.tqdm(range(n_batches + 1)):
        if i == n_batches:
            partial_data_to_be_predict = oot_append.loc[i*nrow_interval:, feas_cols_v2]
        else:
            partial_data_to_be_predict = oot_append.loc[i*nrow_interval: (i+1)*nrow_interval - 1, feas_cols_v2] ## 用loc来切片，“行”这一维度，竟然是上下限都保留！！
        ##################### 
        if ns_prob_col is None:
            ns_prob_col = model.predict(xgb.DMatrix(partial_data_to_be_predict[feas_cols_v2].values))
        else:
            ns_prob_col = np.hstack(
                [
                    ns_prob_col, 
                    model.predict(xgb.DMatrix(partial_data_to_be_predict[feas_cols_v2].values))
                ]
            )
    
    return ns_prob_col

def loadTheData_doThePrediction_v3(
    model_name, feas_to_be_used, df, score_df_path, 
    predict_data_batch = 5, score_col_name = "score", trace_col_name = "trace_id",
    model_dir = f"/data/private/public_data/cpu1/minkexiu/DuoTouSubModels/trained_models",
):
    scores = predict_with_trained_model_v3(
        model_name,
        feas_to_be_used,
        df,
        predict_data_batch,
        model_dir
    )
    
    score_df_dir = os.path.dirname(
        score_df_path
    )
    if not os.path.exists(score_df_dir):
        os.makedirs(score_df_dir)
    
    pd.DataFrame(
        {
            "trace_id": df[trace_col_name],
            score_col_name: scores,
        }
    ).to_csv(score_df_path, index=False)
    print(f"{model_name} prediction done!")

def loadTheData_doThePrediction_v3_cv(
    model_name, feas_to_be_used, df, score_df_path, 
    predict_data_batch = 5, 
    score_col_name = "score", 
    trace_col_name = "trace_id", 
    loanaccound_col_name = "loan_account_id",
    model_dir = f"/data/private/public_data/cpu1/minkexiu/DuoTouSubModels/trained_models",
    cv_num = 5
):
    '''
    支持cross_validation的。
    注意，df里面必须要有loan_account_id啥的。
    '''
    score_list = []
    trace_list = []
    for i in range(cv_num):
        df_tmp = df[df[loanaccound_col_name] % cv_num == i]
        scr = predict_with_trained_model_v3(
            model_name.format(i),
            feas_to_be_used,
            df_tmp,
            predict_data_batch,
            model_dir
        )
        score_list.append(scr)   
        trace_list.append(df_tmp[trace_col_name])
        
    scores = np.hstack(score_list)
    traces = np.hstack(trace_list)
#     print(scores.shape, traces.shape)
    
    score_df_dir = os.path.dirname(
        score_df_path
    )
    if not os.path.exists(score_df_dir):
        os.makedirs(score_df_dir)
    
    pd.DataFrame(
        {
            "trace_id": traces, #df[trace_col_name],
            score_col_name: scores,
        }
    ).to_csv(score_df_path, index=False)
    print(f"{model_name} prediction done!")
    
class Logger(object):
    def __init__(self, file, mode="a"):
        self.terminal = sys.stdout
        self.log = open(file, mode) ## "feixingwei-output-1.txt"

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

        
class redirect_output(object):
    """context manager for reditrecting stdout/err to files"""

    def __init__(self, file, mode="a"):
        directory = os.path.dirname(file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.log = Logger(file, mode)

    def __enter__(self):
        self.sys_stdout = sys.stdout
        self.sys_stderr = sys.stderr
        
        sys.stdout = self.log
        # self.log.close()
        # self.log = Logger()

    def __exit__(self, exc_type, exc_value, traceback):
        self.log.close()
        sys.stdout = self.sys_stdout
        sys.stderr = self.sys_stderr
        
        
def find_best_model_from_log(
    perf_thresh, ## 训练集和验证集表现的最大可接受差距。太大的，就有过拟合嫌疑，就不要了。
    last_trn_perf, ## 上一轮的最好训练集表现。
    last_test_perf, ## 上一轮的最好验证集表现。
    log_file, ## log文件位置。
    first_param, first_range, ## 当前要看的第一个参数，以及参数的范围 
    second_param, second_range ## 当前要看的第二个参数，以及参数的范围 
):
    def find_nearest_val(target_val, origin_list):
        '''
        Find the val in origin_list which is nearest to target_val
        '''
        min_gap = float("inf")
        nearest_val = None
        for v in origin_list:
            distance = abs(v-target_val)
            if distance < min_gap:
                min_gap = distance
                nearest_val = v
        return nearest_val
    with open(log_file, "r") as f:
        lines = f.readlines()
    # print(lines)
    lastline_is_symbolic = False
    cur_model_name = ""
    min_trn_tst_gap = float("inf")
    cur_best_model = ""
    cur_best_perf = (last_trn_perf, last_test_perf)
    cur_best_params = None
    for line in lines:
        if "*****" in line: ## 从log里解析出模型的名字：
            cur_model_name = line.split()[-1]
        if lastline_is_symbolic: ## 找到当前模型最好的早停表现（如果上一行是什么Stopping之类的，那当前行就有最好表现的信息）：
            trnA, tstA = float(re.findall("train-auc:([\.\d]+)", line)[0]), float(re.findall("test-auc:([\.\d]+)", line)[0])
            trn_tst_gap = abs(trnA - tstA)
            if tstA >= cur_best_perf[-1] and trn_tst_gap <= perf_thresh and trn_tst_gap <= min_trn_tst_gap: 
                ## 验证集上表现超越前人，训练集和验证集误差小于可接受的限度，并且这个误差还比前人的都要小。这就是最优的模型。第三个条件可以酌情放松。
                min_trn_tst_gap = trn_tst_gap
                ## 记录当前最好的参数
                first_part, second_part = cur_model_name.split("-")[1:]
                _1, _2 = float(first_part.replace(first_param + "_", "")), float(second_part.replace(second_param + "_", ""))
                cur_best_params = (find_nearest_val(_1, first_range), find_nearest_val(_2, second_range))
                ## 记录当前最好表现和最好模型名称
                cur_best_model = cur_model_name
                cur_best_perf = (trnA, tstA)
        lastline_is_symbolic = True if "Stopping. Best iteration:" in line else False
    # print("Stop looking for...")
    return cur_best_model, cur_best_perf, cur_best_params

def get_importance_fixed1(
    xgb_model_path,
    feas_cols_v2
):
    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_model_path)
    
    weights = xgb_model.get_score(importance_type="weight")
    gains = xgb_model.get_score(importance_type="gain")
    total_gains = {}
    for fea_avt in weights:
        total_gains[fea_avt] = weights[fea_avt]*gains[fea_avt]
        
    iptc_scores = total_gains ## xgb_model.get_score(importance_type="total_gain")
    importances = []
    for i in range(len(feas_cols_v2)):
        fea_avt = f"f{i}"
        if fea_avt not in iptc_scores:
            continue
        importances.append((feas_cols_v2[i], iptc_scores[fea_avt]))
    importances.sort(key=lambda x: x[1], reverse=True)
    sorted_features = [_[0] for _ in importances]
    
    return importances, sorted_features

def get_full_importance_table(
    xgb_model_path,
    feas_cols_v2,
    sorted_by = "total_gain",
    feaInfoTable = "featureField.csv", ## 这个就是从ETL上下载下来的特征列表咯。
    selectCondition = "(infoTab_.场景 == '复贷特征') & (infoTab_.状态=='可用')"
):
    assert sorted_by in {"total_gain", "weight", "gain"}, f'sorted_by must be in {{"total_gain", "weight", "gain"}}'
    # iptc_scores = total_gains ## xgb_model.get_score(importance_type="total_gain")
    def give_iptc_list(iptc_scores, feas_cols_v2):
        importances = []
        for i in range(len(feas_cols_v2)):
            fea_avt = f"f{i}"
            if fea_avt not in iptc_scores:
                continue
            importances.append((feas_cols_v2[i], iptc_scores[fea_avt]))
        importances.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [_[0] for _ in importances]
        return {k:v for (k, v) in importances}, sorted_features
    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_model_path)
    weights = xgb_model.get_score(importance_type="weight")
    gains = xgb_model.get_score(importance_type="gain")
    total_gains = {}
    for fea_avt in weights:
        total_gains[fea_avt] = weights[fea_avt]*gains[fea_avt]
    iptc_weights, sortedFeas_weights = give_iptc_list(weights, feas_cols_v2)
    iptc_gains, sortedFeas_gains = give_iptc_list(gains, feas_cols_v2)
    iptc_totGains, sortedFeas_totGains = give_iptc_list(total_gains, feas_cols_v2)
    iptc_df = pd.DataFrame({
        "feature": sortedFeas_weights,
        "weight": [iptc_weights[k] for k in sortedFeas_weights],
        "gain": [iptc_gains[k] for k in sortedFeas_weights],
        "total_gain": [iptc_totGains[k] for k in sortedFeas_weights],
    })
    iptc_df = iptc_df.sort_values(by = [sorted_by,],ascending=False).reset_index(drop=True)
    
    infoTab_ = pd.read_table(feaInfoTable)
    infoTab = infoTab_[eval(selectCondition)]
    infoTab["feature"] = infoTab.名称.str.lower()
    
    return pd.merge(
        iptc_df, infoTab[["feature", "描述"]], on="feature", how="inner"
    )

def get_full_importance_noTable(
    xgb_model_path,
    feas_cols_v2,
    sorted_by = "total_gain",
    feaInfoTable = "featureField.csv", ## 这个已经没用了。
    selectCondition = "(infoTab_.场景 == '复贷特征') & (infoTab_.状态=='可用')" ## 也没用了。
):
    assert sorted_by in {"total_gain", "weight", "gain"}, f'sorted_by must be in {{"total_gain", "weight", "gain"}}'
    # iptc_scores = total_gains ## xgb_model.get_score(importance_type="total_gain")
    def give_iptc_list(iptc_scores, feas_cols_v2):
        importances = []
        for i in range(len(feas_cols_v2)):
            fea_avt = f"f{i}"
            if fea_avt not in iptc_scores:
                continue
            importances.append((feas_cols_v2[i], iptc_scores[fea_avt]))
        importances.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [_[0] for _ in importances]
        return {k:v for (k, v) in importances}, sorted_features
    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_model_path)
    weights = xgb_model.get_score(importance_type="weight")
    gains = xgb_model.get_score(importance_type="gain")
    total_gains = {}
    for fea_avt in weights:
        total_gains[fea_avt] = weights[fea_avt]*gains[fea_avt]
    iptc_weights, sortedFeas_weights = give_iptc_list(weights, feas_cols_v2)
    iptc_gains, sortedFeas_gains = give_iptc_list(gains, feas_cols_v2)
    iptc_totGains, sortedFeas_totGains = give_iptc_list(total_gains, feas_cols_v2)
    iptc_df = pd.DataFrame({
        "feature": sortedFeas_weights,
        "weight": [iptc_weights[k] for k in sortedFeas_weights],
        "gain": [iptc_gains[k] for k in sortedFeas_weights],
        "total_gain": [iptc_totGains[k] for k in sortedFeas_weights],
    })
    iptc_df = iptc_df.sort_values(by = [sorted_by,],ascending=False).reset_index(drop=True)
    return iptc_df

#     infoTab_ = pd.read_table(feaInfoTable)
#     infoTab = infoTab_[eval(selectCondition)]
#     infoTab["feature"] = infoTab.名称.str.lower()
#     return pd.merge(
#         iptc_df, infoTab[["feature", "描述"]], on="feature", how="inner"
#     )

def csv2Parquet(csv_filePath, pqt_filePath, batch_size = 10*10000, fileType = "csv"):
    '''
    读csv的时候，每次读入最多batch_size行，然后写入parquet中。
    可以记录载入数据的进度情况，这样在载入大数据的时候不会两眼一抹黑。
    '''    
    assert fileType in ["csv", "table"], "fileType should be in `csv` or `table`."
    
    pqt_dir = os.path.split(pqt_filePath)[0]
    if not os.path.exists(pqt_dir):
        os.makedirs(pqt_dir)
    
    n = 0
    writer = None
    schema = None
    if fileType == "csv":
        df = pd.read_csv(csv_filePath, chunksize=batch_size)
    elif fileType == "table":
        df = pd.read_table(csv_filePath, chunksize=batch_size)
    else:
        raise Exception("~~~something is wrong~~~")
    for chunk in tqdm.tqdm(df):
        if n == 0:
            table = pa.Table.from_pandas(chunk)
            writer = pq.ParquetWriter(pqt_filePath, table.schema)
            schema = table.schema
        else:
            # for colName in table.schema.names:
            table = pa.Table.from_pandas(chunk, schema=schema)

        writer.write_table(table=table)
        n += 1
    if writer: ## 这里就是卡了我很久的关键。
        writer.close()
        
        
def finetune_2_params(
    all_train_matrix, 
    all_valid_matrix,
    model_dir_this_round, 
    model_name, 
    params,
    first_param = "max_depth",
    first_range = [3, 5, ], 
    first_type = "int",
    second_param = "min_child_weight",
    second_range = [100, 1000, ],
    second_type = "int",
):
    '''
    这个函数是固定调两个参数的，一定要调2个参数。多了少了都不行。
    调first_param和second_param俩参数，最终返回修改过后的参数组合以及最佳树数。
    '''
    
    ## 原版params里面的相应参数，如果不在调参列表里，也要加进去，作为base。
    if not (params[first_param] in first_range):
        first_range = [params[first_param]] + first_range
    if not (params[second_param] in second_range):
        second_range = [params[second_param]] + second_range
    
    
    ## 存模型表现的文件的位置。
    record_file = os.path.join(
        model_dir_this_round, model_name
    ) + f"-finetuning-{first_param}-{second_param}.csv"
    ## 如果这个文件不存在，就硬造一个空的文件。
    if not os.path.exists(record_file):
        pd.DataFrame(
            {
                "param_setting": [],
                "best_score": [],
                "best_iteration": []
            }

        ).to_csv(
            record_file, index=False
        )
    ## 初始化一个新的参数列表。
    params_thisRound = {key:params[key] for key in params}
    ## 调两个参数。
    for v1_ in first_range: 
        for v2_ in second_range: 

            v1 = eval(f"{first_type}({v1_})")
            v2 = eval(f"{second_type}({v2_})")
            
            params_thisRound[first_param] = v1
            params_thisRound[second_param] = v2

            records = pd.read_csv(record_file)
            param_setting = f"{first_param}__{v1}-{second_param}__{v2}"
            if param_setting in records.param_setting.to_list():
                print(f"{param_setting} is already trained.".center(100, "="))
            else:
                print(f"{param_setting} is now training...".center(100, "="))
                md = train_model_with_different_label_3(
                    all_train_matrix, all_valid_matrix, params_thisRound, 
                    model_name, 
                    save_model_path = model_dir_this_round,
                    save_model = False 
                )
                records = records.append(
                    pd.DataFrame({
                        "param_setting": [param_setting],
                        "best_score": [md.best_score],
                        "best_iteration": [md.best_iteration],
                    })
                )
                records.to_csv(record_file, index=False)
            print() 
    ## 寻找最佳参数：
    records = pd.read_csv(record_file)
    best_iteration = records.iloc[records.best_score.idxmax(), :]["best_iteration"]
    best_params = records.iloc[records.best_score.idxmax(), :]["param_setting"]
    best_pm_dict = {pt.split("__")[0]: pt.split("__")[-1] for pt in best_params.split("-")}
    params_thisRound[first_param] = eval(f"{first_type}({best_pm_dict[first_param]})")
    params_thisRound[second_param] = eval(f"{second_type}({best_pm_dict[second_param]})")
    return params_thisRound, best_iteration

def loadTheData_doThePrediction_v3_cv_lr(
    model_name, feas_to_be_used, df, score_df_path, 
    predict_data_batch = 5, 
    score_col_name = "score", 
    trace_col_name = "trace_id", 
    loanaccound_col_name = "loan_account_id",
    model_dir = f"/data/private/public_data/cpu1/minkexiu/DuoTouSubModels/trained_models",
    cv_num = 5
):
    '''
    支持cross_validation的。
    注意，df里面必须要有loan_account_id啥的。
    '''
    score_list = []
    trace_list = []
    for i in range(cv_num):
        df_tmp = df[df[loanaccound_col_name] % cv_num == i]
        
        with open(os.path.join(model_dir, model_name.format(i)), 'rb') as file:
            best_lr_model = pickle.load(file)   
        with open(os.path.join(model_dir, model_name.format(i) + "__feaValScaler"), 'rb') as file:
            standar_scaler = pickle.load(file)   
        scr = best_lr_model.predict_proba(
            pd.DataFrame(standar_scaler.transform(df_tmp[feas_to_be_used]), columns=feas_to_be_used)
        )[:,1]
        score_list.append(scr)   
        trace_list.append(df_tmp[trace_col_name])
        
    scores = np.hstack(score_list)
    traces = np.hstack(trace_list)
    
    score_df_dir = os.path.dirname(
        score_df_path
    )
    if not os.path.exists(score_df_dir):
        os.makedirs(score_df_dir)
    
    pd.DataFrame(
        {
            trace_col_name: traces, #df[trace_col_name],
            score_col_name: scores,
        }
    ).to_csv(score_df_path, index=False)
    print(f"{model_name} prediction done!")