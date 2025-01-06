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

# import lightgbm as lgb

# from perf_eval_tools import *

def output_as_fig(text):
    '''
    将text变成一张图片，然后输出。
    这么做没有什么特别用意，目的就在于能够让输出变得好看一些。
    '''
    # 创建一张空白的图片
    img = Image.new('RGB', (500, 200), color = (255, 255, 255))
    
    # 在图片中添加文本
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('~/SimHei.ttf', 36)
    textwidth, textheight = draw.textsize(text, font)

    x = (img.width - textwidth) / 2
    y = (img.height - textheight) / 2

    draw.text((x, y), text, font=font, fill=(0, 0, 0))

    # 在 Jupyter Notebook 中显示图片
#     imshow(img)
    img.save('text_image.png')
    display(Image.open('text_image.png'))

# def bin_by_scr_and_see_feaVal_distribution_1(
#     cross,
#     scr_col,
#     labels, 
#     feas_figure,
#     convert_nBins = 10,
#     color_range = [0,1]
# ):
#     '''
#     cross是一个表，把评分和特征都整合在一起的表。
#     根据scr_col这个分数进行分bin。
#     然后在各个分bin里统计feas_figure这些特征的一些统计值："平均数", "最小值", "最大值", "众数"。
#     这个的结果可以视作用户画像。
#     '''
#     try: 
#         tbs = []
#         format_dict = {colName: '{:.3f}' for colName in feas_figure}
#         bin_col_name = f"{scr_col}-bin"
#         cross.loc[:, bin_col_name] = pd.qcut(
#             cross[scr_col], q=convert_nBins, # retbins=True, labels=False
#             duplicates="drop", ## 这个设置了之后，重复边界就会被删掉。
#         )
#         group_new = cross.groupby(bin_col_name)

#         for calc_type, calc_name in zip(
#             ["mean()", ],#"min()", "max()", "apply(lambda x:x.mode())"
#             ["平均数", ]#"最小值", "最大值", "众数"
#         ): ## 我们做这个是为了做一些不一样的统计。
#             rst = {}
#             actual_col_names = []
#             for fea in labels + feas_figure:
#                 rst[fea] = eval(f"group_new[fea].{calc_type}")
#                 actual_col_names.append(fea)
#                 if fea in labels:
#                     rst[f"{fea}-lift"] = rst[fea] / cross[fea].mean()
#                     actual_col_names.append(f"{fea}-lift")
#             rst["bin_size"] = group_new[scr_col].count()
            
#             df__ = pd.DataFrame(rst)[["bin_size"] + actual_col_names].reset_index().rename(
#                 ## 把名字很长的列改一下名字。
#                 columns = {
#                     "riskorderrenewfeature___riskonloanorderfeature___unpaidinstalmentprincipal": "在贷订单___风控时刻剩余未结清本金",
#                     "bairongmultiloanfeature___m3___id___nbank___orgnum": "近三个月多头机构申请次数",
#                     "bairongmultiloanfeature___m3___id___nbank___allnum": "近三个月多头申请次数",
#                     "general_applyorderage": "完件年龄",
#                 }
#             )
            
#             tbs.append(
#                 df__
#                 .style
# #                 .format(format_dict)
#                 .background_gradient(
#                     axis=0, cmap="YlOrRd", low=color_range[0], high=color_range[1], subset = df__.columns[2:]#feas_figure
#                 ).set_caption(f"分bin后各bin的：{calc_name}")
#             )
#         display_side_by_side(tbs)
#     except Exception as e:
# #         output_as_fig(f"{scr_col}\n有问题")
#         print(e)
    
def bin_by_scr_and_see_feaVal_distribution_1(
    cross,
    scr_col,
    labels, 
    feas_figure,
    convert_nBins = 10,
    color_range = [0,1],
    scr_col_bin = True,
):
    '''
    cross是一个表，把评分和特征都整合在一起的表。
    根据scr_col这个分数进行分bin。
    然后在各个分bin里统计feas_figure这些特征的一些统计值："平均数", "最小值", "最大值", "众数"。
    这个的结果可以视作用户画像。
    scr_col_bin_or_not 这个参数如果为True，就要对scr_col这个列进行分bin；如果scr_col列本来就是分bin的形式，那就设为false，就不需要再分bin了。
    '''
    try: 
        tbs = []
        format_dict = {colName: '{:.3f}' for colName in feas_figure}
        bin_col_name = f"{scr_col}-bin"
        if scr_col_bin:
            cross.loc[:, bin_col_name] = pd.qcut(
                cross[scr_col], q=convert_nBins, # retbins=True, labels=False
                duplicates="drop", ## 这个设置了之后，重复边界就会被删掉。
            )
        else:
            cross.loc[:, bin_col_name] = cross[scr_col]
        group_new = cross.groupby(bin_col_name)

        for calc_type, calc_name in zip(
            ["mean()", ],#"min()", "max()", "apply(lambda x:x.mode())"
            ["平均数", ]#"最小值", "最大值", "众数"
        ): ## 我们做这个是为了做一些不一样的统计。
            rst = {}
            actual_col_names = []
            for fea in labels + feas_figure:
                rst[fea] = eval(f"group_new[fea].{calc_type}")
                actual_col_names.append(fea)
                if fea in labels:
                    rst[f"{fea}-lift"] = rst[fea] / cross[fea].mean()
                    actual_col_names.append(f"{fea}-lift")
            rst["bin_size"] = group_new[scr_col].count()
            
            df__ = pd.DataFrame(rst)[["bin_size"] + actual_col_names].reset_index().rename(
                ## 把名字很长的列改一下名字。
                columns = {
                    "riskorderrenewfeature___riskonloanorderfeature___unpaidinstalmentprincipal": "在贷订单___风控时刻剩余未结清本金",
#                     "bairongmultiloanfeature___m3___id___nbank___orgnum": "近三个月多头机构申请次数",
#                     "bairongmultiloanfeature___m3___id___nbank___allnum": "近三个月多头申请次数",
    "bairongmultiloanfeature___m3___id___nbank___orgnum": "近3个月非银_机构申请次数",
    "bairongmultiloanfeature___m6___id___nbank___orgnum": "近6个月非银_机构申请次数",
    "bairongmultiloanfeature___m12___id___nbank___orgnum": "近12个月非银_机构申请次数",
    "bairongmultiloanfeature___m3___id___nbank___allnum": "近3个月非银_申请次数",
    "bairongmultiloanfeature___m6___id___nbank___allnum": "近6个月非银_申请次数",
    "bairongmultiloanfeature___m12___id___nbank___allnum": "近12个月非银_申请次数",
                    "general_applyorderage": "申请年龄",
                }
            )
            
            tbs.append(
                df__
                .style
#                 .format(format_dict)
                .background_gradient(
                    axis=0, cmap="YlOrRd", low=color_range[0], high=color_range[1], subset = df__.columns[2:]#feas_figure
                ).set_caption(f"分bin后各bin的：{calc_name}")
            )
        display_side_by_side(tbs)
    except Exception as e:
#         output_as_fig(f"{scr_col}\n有问题")
        print(e)