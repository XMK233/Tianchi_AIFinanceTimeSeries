import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import time
import xgboost as xgb
import os, time
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import sklearn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import tqdm, re, pickle, gc, os, time, random, sys

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from collections import defaultdict

import json
from pprint import pprint

pd.set_option('display.max_columns', None)

import deepdiff, glob
import math

from sklearn.model_selection import GridSearchCV,ParameterGrid,StratifiedKFold,train_test_split,GroupKFold
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, ElasticNet, ElasticNetCV, LogisticRegressionCV

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from IPython.display import display
from matplotlib.pyplot import imshow

import lightgbm as lgb
def train_model_with_different_label_3_lgbm(
    dtrain, dtest, params, 
    saved_model_name, 
    save_model_path = "/data/private/public_data/cpu1/minkexiu/DuoTouSubModels/trained_models",
    save_model = True,
    early_stopping_rounds = 200, 
):
    callbacks = [lgb.log_evaluation(period = 500), lgb.early_stopping(stopping_rounds = early_stopping_rounds)]
    train_start_time = time.time()
    booster_maidian = lgb.train(
        params = params,
        train_set = dtrain,
        num_boost_round = 100000,
        valid_sets = [dtrain, dtest],
        valid_names = ['train', 'eval'],
        callbacks = callbacks,
        # verbose_eval=False
    )
    train_end_time = time.time()
    
    print(f"train time: {train_end_time - train_start_time}")
        
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    if save_model:
        booster_maidian.save_model(os.path.join(save_model_path, saved_model_name))
        
    return booster_maidian

def train_model_with_different_label_2_variousParam_nbr_lgbm(
    dtrain, dtest, saved_model_name, 
    do_valid = True, save_model_path = "/data/private/public_data/cpu1/minkexiu/DuoTouSubModels/trained_models", 
    params = None, nbr = 100, ## 这个就可以用来设置模型的树的数量
    early_stopping_rounds = 200, 
):
    #############################
    if params is None:
        params = {
            "boosting_type": 'gbdt',
            "objective": "binary",
            "boosting_type": 'gbdt',
            'scale_pos_weight': negNum_posNum_ratio,
            "metric": "auc",
            "device_type": "gpu",
            "learning_rate": 0.1,
            "n_estimators": 100000,

            ## 第一轮调参：
            "max_depth":3,
            "min_child_weight": 100,
            ## 第二轮调参：
            "colsample_bytree": 1,
            "subsample": 1,
            ## 第三轮调参：
            'reg_lambda': 7,
            'reg_alpha': 100,  
            
            'num_leaves': 4,
            "nthread": 25,
            "verbose": -1,
        }
    ############################
    params["n_estimators"] = nbr
    print(params)
    callbacks = [lgb.log_evaluation(period = 500), lgb.early_stopping(stopping_rounds = early_stopping_rounds)]
    train_start_time = time.time()
        
    vs = [dtrain, dtest] if do_valid else [dtrain, ]
    vn = ['train', 'eval'] if do_valid else ["train", ]
    # print(vs, vn, nbr)
    booster_maidian = lgb.train(
        params = params,
        train_set = dtrain,
        # num_boost_round = 10, ## 这个参数我设置了之后似乎不生效。
        valid_sets = vs,
        valid_names = vn,
        callbacks = callbacks,
        # verbose_eval=False
    )
        
    train_end_time = time.time()
    
    print(f"train time: {train_end_time - train_start_time}")
    
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    booster_maidian.save_model(os.path.join(save_model_path, saved_model_name))
    
    return booster_maidian

def get_full_importance_noTable_lgbm(
    lgb_model_path,
    feas_cols_v2,
    sorted_by = "gain",
    ignore_no_impc_feas = True, ## 这个参数设置为True之后，会把sorted_by对应的重要性为0的特征从最后返回的iptc_df中去除。
):
    assert sorted_by in {"split", "gain"}, f'sorted_by must be in {{"split", "gain"}}'
    
    lgb_model = lgb.Booster(model_file = lgb_model_path)
    splits = lgb_model.feature_importance("split")
    gains = lgb_model.feature_importance("gain")
    
    iptc_df = pd.DataFrame({
        "feature": feas_cols_v2,
        "split": lgb_model.feature_importance("split"),
        "gain": lgb_model.feature_importance("gain"),
    })

    iptc_df = iptc_df.sort_values(by = [sorted_by,],ascending=False).reset_index(drop=True)
    return iptc_df if not ignore_no_impc_feas else iptc_df[iptc_df[sorted_by] > 0]

def finetune_2_params_lgbm(
    all_train_matrix, 
    all_valid_matrix,
    model_dir_this_round, 
    model_name, 
    params, ## 基础参数。
    first_param = "max_depth",
    first_range = [3, 5, ], 
    first_type = "int",
    second_param = "min_child_weight",
    second_range = [100, 1000, ],
    second_type = "int",
    record_file_identifier = "",
    early_stopping_rounds = 200, 
    max_trn_evl_diff = 0.1, ## 这个是啥呢？就是train和eval的auc之间的差值。如果大于这个阈值，说明有过拟合嫌疑，这种结果不予记录。
):
    '''
    这个函数是固定调两个参数的，一定要调2个参数。多了少了都不行。
    调first_param和second_param俩参数，最终返回修改过后的参数组合以及最佳树数。
    record_file_identifier 这个参数是用来单独标记模型表现记录文件的。
    '''
    
    ## 原版params里面的相应参数，如果不在调参列表里，也要加进去，作为base。
    if not (params[first_param] in first_range):
        first_range = [params[first_param]] + first_range
    if not (params[second_param] in second_range):
        second_range = [params[second_param]] + second_range
    
    
    ## 存模型表现的文件的位置。
    record_file = os.path.join(
        model_dir_this_round, model_name
    ) + f"-finetuning-{first_param}-{second_param}{record_file_identifier}.csv"
    ## 如果这个文件不存在，就硬造一个空的文件。
    if not os.path.exists(record_file):
        pd.DataFrame(
            {
                "param_setting": [],
                "train_auc": [],
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
            if second_type == "str":
                v2 = eval(f"{second_type}('{v2_}')")
            else:
                v2 = eval(f"{second_type}({v2_})")
            
            params_thisRound[first_param] = v1
            params_thisRound[second_param] = v2

            records = pd.read_csv(record_file)
            param_setting = f"{first_param}__{v1}-{second_param}__{v2}"
            if param_setting in records.param_setting.to_list():
                display(records[records.param_setting == param_setting])
                # print(f"{param_setting} is already trained.".center(100, "="))
            else:
                print(f"{param_setting} is now training...".center(100, "="))
                md = train_model_with_different_label_3_lgbm(
                    all_train_matrix, all_valid_matrix, params_thisRound, 
                    model_name, 
                    save_model_path = model_dir_this_round,
                    save_model = False,
                    early_stopping_rounds = early_stopping_rounds
                )
                records = records.append(
                    pd.DataFrame({
                        "param_setting": [param_setting], 
                        "train_auc": md.best_score["train"]["auc"], ## 训练集的AUC情况。
                        "best_score": md.best_score["eval"]["auc"], ## 验证集的AUC情况。
                        "best_iteration": [md.best_iteration],
                    })
                )
                records.to_csv(record_file, index=False)
            print() 
    ## 寻找最佳参数：
    records = pd.read_csv(record_file)
    records = records[ 
        (records.train_auc - records.best_score).abs() <= max_trn_evl_diff
    ].reset_index(drop=True) ## 训练集表现和验证集表现之差，不得超过阈值。
    best_iteration = records.iloc[records.best_score.idxmax(), :]["best_iteration"]
    best_params = records.iloc[records.best_score.idxmax(), :]["param_setting"]
    best_pm_dict = {pt.split("__")[0]: pt.split("__")[-1] for pt in best_params.split("-")}
    params_thisRound[first_param] = eval(f"{first_type}({best_pm_dict[first_param]})")
    params_thisRound[second_param] = eval(f"{second_type}({best_pm_dict[second_param]})")
    return params_thisRound, best_iteration

def finetune_2_params_lgbm_1(
    all_train_matrix, 
    all_valid_matrix,
    model_dir_this_round, 
    model_name, 
    params, ## 基础参数。
    iternums, ## 原始的树棵树。
    first_param = "max_depth",
    first_range = [3, 5, ], 
    first_type = "int",
    second_param = "min_child_weight",
    second_range = [100, 1000, ],
    second_type = "int",
    record_file_identifier = "",
    early_stopping_rounds = 200, 
    max_trn_evl_diff = 0.1, ## 这个是啥呢？就是train和eval的auc之间的差值。如果大于这个阈值，说明有过拟合嫌疑，这种结果不予记录。
):
    '''
    这个函数是固定调两个参数的，一定要调2个参数。多了少了都不行。
    调first_param和second_param俩参数，最终返回修改过后的参数组合以及最佳树数。
    record_file_identifier 这个参数是用来单独标记模型表现记录文件的。
    
    相比于 finetune_2_params_lgbm 版本，加了一个参数 iternums。如果所有的参数组合，都有过拟合倾向（train和eval的AUC相差大于max_trn_evl_diff），那么所有的参数组合都不合理，那就原样输出params和iternums。
    '''
    
    ## 原版params里面的相应参数，如果不在调参列表里，也要加进去，作为base。
    if not (params[first_param] in first_range):
        first_range = [params[first_param]] + first_range
    if not (params[second_param] in second_range):
        second_range = [params[second_param]] + second_range
    
    
    ## 存模型表现的文件的位置。
    record_file = os.path.join(
        model_dir_this_round, model_name
    ) + f"-finetuning-{first_param}-{second_param}{record_file_identifier}.csv"
    ## 如果这个文件不存在，就硬造一个空的文件。
    if not os.path.exists(record_file):
        pd.DataFrame(
            {
                "param_setting": [],
                "train_auc": [],
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
            
            if first_type == "str":
                v1 = eval(f"{first_type}('{v1_}')")
            else:
                v1 = eval(f"{first_type}({v1_})")
            if second_type == "str":
                v2 = eval(f"{second_type}('{v2_}')")
            else:
                v2 = eval(f"{second_type}({v2_})")
            
            params_thisRound[first_param] = v1
            params_thisRound[second_param] = v2

            records = pd.read_csv(record_file)
            param_setting = f"{first_param}__{v1}-{second_param}__{v2}"
            if param_setting in records.param_setting.to_list():
                display(records[records.param_setting == param_setting])
                # print(f"{param_setting} is already trained.".center(100, "="))
            else:
                print(f"{param_setting} is now training...".center(100, "="))
                md = train_model_with_different_label_3_lgbm(
                    all_train_matrix, all_valid_matrix, params_thisRound, 
                    model_name, 
                    save_model_path = model_dir_this_round,
                    save_model = False,
                    early_stopping_rounds = early_stopping_rounds
                )
                records = records.append(
                    pd.DataFrame({
                        "param_setting": [param_setting], 
                        "train_auc": md.best_score["train"]["auc"], ## 训练集的AUC情况。
                        "best_score": md.best_score["eval"]["auc"], ## 验证集的AUC情况。
                        "best_iteration": [md.best_iteration],
                    })
                )
                records.to_csv(record_file, index=False)
            print() 
    ## 寻找最佳参数：
    records = pd.read_csv(record_file)
    records = records[ 
        (records.train_auc - records.best_score).abs() <= max_trn_evl_diff
    ].reset_index(drop=True) ## 训练集表现和验证集表现之差，不得超过阈值。
    if len(records) <= 0:
        print(f"all possible combinations have train-eval gap larger than {max_trn_evl_diff}, returning original params and tree num")
        return params, iternums
    else:
        best_iteration = records.iloc[records.best_score.idxmax(), :]["best_iteration"]
        best_params = records.iloc[records.best_score.idxmax(), :]["param_setting"]
        best_pm_dict = {pt.split("__")[0]: pt.split("__")[-1] for pt in best_params.split("-")}
        if first_type == "str":
            params_thisRound[first_param] = eval(f"{first_type}('{best_pm_dict[first_param]}')")
        else:
            params_thisRound[first_param] = eval(f"{first_type}({best_pm_dict[first_param]})")
        if second_type == "str":
            params_thisRound[second_param] = eval(f"{second_type}('{best_pm_dict[second_param]}')")
        else:
            params_thisRound[second_param] = eval(f"{second_type}({best_pm_dict[second_param]})")
        return params_thisRound, best_iteration

def predict_with_trained_model_v3_lgbm(
    model_name, feas_to_be_used, oot_append, 
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
    model = lgb.Booster(model_file = os.path.join(model_dir, model_name))
    
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
            ns_prob_col = model.predict(
                partial_data_to_be_predict[feas_cols_v2]
#                 lgb.Dataset(
#                     partial_data_to_be_predict[feas_cols_v2], 
#                     feature_name=feas_cols_v2,
#                     free_raw_data=False
#                 )
            )
        else:
            ns_prob_col = np.hstack(
                [
                    ns_prob_col, 
                    model.predict(
                        partial_data_to_be_predict[feas_cols_v2]
#                         lgb.Dataset(
#                             partial_data_to_be_predict[feas_cols_v2], 
#                             feature_name=feas_cols_v2,
#                             free_raw_data=False
#                         )
                    )
                ]
            )
    
    return ns_prob_col

def loadTheData_doThePrediction_v3_lgbm(
    model_name, feas_to_be_used, df, score_df_path, 
    predict_data_batch = 5, score_col_name = "score", trace_col_name = "trace_id",
    model_dir = f"/data/private/public_data/cpu1/minkexiu/DuoTouSubModels/trained_models",
):
    scores = predict_with_trained_model_v3_lgbm(
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
    
def loadTheData_doThePrediction_v3_cv_lgbm(
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
        df_tmp = df[df[loanaccound_col_name] % cv_num == i].reset_index(drop=True)
        scr = predict_with_trained_model_v3_lgbm(
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
            trace_col_name: traces, 
            score_col_name: scores,
        }
    ).to_csv(score_df_path, index=False)
    print(f"{model_name} prediction done!")
    
def finetune_2_params_lgbm_smallData(
    all_trnVld_matrix, 
    model_dir_this_round, 
    model_name, 
    params, ## 基础参数。
    first_param = "max_depth",
    first_range = [3, 5, ], 
    first_type = "int",
    second_param = "min_child_weight",
    second_range = [100, 1000, ],
    second_type = "int",
    record_file_identifier = "",
    early_stopping_rounds = 200, 
):
    '''
    这个函数是固定调两个参数的，一定要调2个参数。多了少了都不行。
    调first_param和second_param俩参数，最终返回修改过后的参数组合以及最佳树数。
    record_file_identifier 这个参数是用来单独标记模型表现记录文件的。
    
    这个仅限于小规模样本的调参。因为我们用了lgb.cv来获得某种参数的表现。
    如果是大规模数据调参，这种方法效率堪忧。
    '''
    
    ## 原版params里面的相应参数，如果不在调参列表里，也要加进去，作为base。
    if not (params[first_param] in first_range):
        first_range = [params[first_param]] + first_range
    if not (params[second_param] in second_range):
        second_range = [params[second_param]] + second_range
    
    
    ## 存模型表现的文件的位置。
    record_file = os.path.join(
        model_dir_this_round, model_name
    ) + f"-finetuning-{first_param}-{second_param}{record_file_identifier}.csv"
    ## 如果这个文件不存在，就硬造一个空的文件。
    if not os.path.exists(record_file):
        pd.DataFrame(
            {
                "param_setting": [],
                "train_auc": [],
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
#                 print(f"{param_setting} is already trained.".center(100, "="))
                display(records[records.param_setting == param_setting])
            else:
                print(f"{param_setting} is now training...".center(100, "="))
                res = lgb.cv(
                    params_thisRound, all_trnVld_matrix, nfold=5,
                    metrics = "auc",
                    early_stopping_rounds = early_stopping_rounds,
                    verbose_eval = 500,
                    eval_train_metric = True
                )
                for idxx, (ta, va) in enumerate(zip(res["train auc-mean"], res["valid auc-mean"])):
                    if ((idxx + 1)%20) == 0:
                        print(f"[{idxx + 1}] train auc-mean: {ta}, valid auc-mean: {va}")
                print("the last perf: ")
                print("[{}] train auc-mean: {}, valid auc-mean: {}".format((idxx + 1), res["train auc-mean"][-1], res["valid auc-mean"][-1]))
                records = records.append(
                    pd.DataFrame({
                        "param_setting": [param_setting], 
                        "train_auc": [res["train auc-mean"][-1]],
                        "best_score": [res["valid auc-mean"][-1]], # md.best_score["eval"]["auc"], 
                        "best_iteration": [len(res["valid auc-mean"])],
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