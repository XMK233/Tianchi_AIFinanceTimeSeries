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
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# import numpy as np
# import seaborn as sns; sns.set()

# import sklearn
# import numpy as np
# from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# import sklearn.preprocessing as pre_processing

# import tqdm, re, pickle, gc, os, time, random, glob

# import warnings
# warnings.filterwarnings("ignore")

# from sklearn.model_selection import train_test_split

# from collections import defaultdict
# from sklearn.preprocessing import KBinsDiscretizer

# from sklearn.model_selection import GridSearchCV,ParameterGrid,StratifiedKFold,train_test_split,GroupKFold
# from sklearn import preprocessing
# from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, ElasticNet, ElasticNetCV, LogisticRegressionCV

# from PIL import Image
# from PIL import ImageDraw
# from PIL import ImageFont
# from IPython.display import display
# from matplotlib.pyplot import imshow


from matplotlib.font_manager import FontProperties
############ 累积曲线系列 ################
def pointwise_accumu_line(
    df,
    unpaid,
    principal,
    title,
    score_list,
    model_name_list,
    p = 0,
    n_bins = 20
):
    '''
    绘制两个图，一个是分箱之后的累积转化率，另一个是按照评分排序之后的逐个样本计算的累积转化率。
    参数解析：
        df: 原来的数据表格。
        unpaid: 说白了就是分子，比如原始label。
        principal: 说白了就是分母，比如一列全都是1的列。请在调用这个函数之前，在df里面加入这一列。
        title: 图的title。
        score_list: 说白了就是预测评分的列名。
        model_name_list: 说白了就是图例里面各列的名字。出于简单可以就直接跟列名一样。
        p: 默认是0，说明这个折线从横轴为0的地方开始画；这个值的取值范围是[0,1)，就是从百分之多少开始绘图。有的时候，折线开头会很高，那么
我们就可以用这个参数跳过前面的部分。
    '''
    result = df
    result.reset_index(drop = True, inplace = True)   
    
    ################
    def getkey(element):
        return element[0]

    def draw_bad_debt_rate(pred_result):
        pred_result.sort(key = getkey)
        x = []
        y = []
        length = len(pred_result)
        principal_total = 0.
        bad_debt_total = 0.
        for i in range(length):
            x.append((i + 1.0)/length)
            principal_total = principal_total + pred_result[i][1]
            bad_debt_total = bad_debt_total + pred_result[i][2]
            y.append(bad_debt_total / principal_total)
        return x, y
    
    
    ##cumulative
    result_vintage = []
    for i in range(len(score_list)):
        # print(score_list[i], result[score_list[i]].isnull().sum())
        model_x, model_y = draw_bad_debt_rate(
            list(
                zip(
                    result[score_list[i]], ## 把某一评分列拿出来
                    result[principal], ## 全为1的列
                    result[unpaid] ## label
                )
            )
        )
        model_vintage = pd.DataFrame({'pass_rate': model_x, 'cum_vintage': model_y, 'model': str(i + 1)})
        result_vintage.append(model_vintage)
        
    result_vintage = pd.concat(result_vintage)
    ################
    plt.figure(figsize=(10, 5), dpi = 150)
    ax = sns.lineplot(
        x="pass_rate", 
        y="cum_vintage", 
        hue="model", 
        data=result_vintage[result_vintage['pass_rate'] >= p].reset_index(drop = True),
        linewidth = 0.8
    )
    ax.set_title(title)
    h, l = ax.get_legend_handles_labels() ## https://stackoverflow.com/questions/53734332/seaborn-barplot-legend-labels-lose-color
    ax.legend(h, model_name_list, title = "Model")
    plt.xlabel('Percentile')
    plt.ylabel('Cumulative Bad Debt Rate')
    plt.legend(model_name_list)
    plt.show()
    
def pointwise_accumu_line_1(
    df,
    unpaid,
    principal,
    title,
    score_list,
    model_name_list,
    p = 0,
    n_bins = 20
):
    '''
    绘制两个图，一个是分箱之后的累积转化率，另一个是按照评分排序之后的逐个样本计算的累积转化率。
    参数解析：
        df: 原来的数据表格。
        unpaid: 说白了就是分子，比如原始label。
        principal: 说白了就是分母，比如一列全都是1的列。相比于pointwise_accumu_line的改进是嘛呢，就是ho，df里这一列不用在外面加，可以在函数内加了。
        title: 图的title。
        score_list: 说白了就是预测评分的列名。
        model_name_list: 说白了就是图例里面各列的名字。出于简单可以就直接跟列名一样。
        p: 默认是0，说明这个折线从横轴为0的地方开始画；这个值的取值范围是[0,1)，就是从百分之多少开始绘图。有的时候，折线开头会很高，那么
我们就可以用这个参数跳过前面的部分。
    '''
    df[principal] = 1 ## 这里是相较于pointwise_accumu_line的改进。
    result = df
    result.reset_index(drop = True, inplace = True)   
    
    ################
    def getkey(element):
        return element[0]

    def draw_bad_debt_rate(pred_result):
        pred_result.sort(key = getkey)
        x = []
        y = []
        length = len(pred_result)
        principal_total = 0.
        bad_debt_total = 0.
        for i in range(length):
            x.append((i + 1.0)/length)
            principal_total = principal_total + pred_result[i][1]
            bad_debt_total = bad_debt_total + pred_result[i][2]
            y.append(bad_debt_total / principal_total)
        return x, y
    
    
    ##cumulative
    result_vintage = []
    for i in range(len(score_list)):
        # print(score_list[i], result[score_list[i]].isnull().sum())
        model_x, model_y = draw_bad_debt_rate(
            list(
                zip(
                    result[score_list[i]], ## 把某一评分列拿出来
                    result[principal], ## 全为1的列
                    result[unpaid] ## label
                )
            )
        )
        model_vintage = pd.DataFrame({'pass_rate': model_x, 'cum_vintage': model_y, 'model': str(i + 1)})
        result_vintage.append(model_vintage)
        
    result_vintage = pd.concat(result_vintage)
    ################
    plt.figure(figsize=(10, 5), dpi = 150)
    ax = sns.lineplot(
        x="pass_rate", 
        y="cum_vintage", 
        hue="model", 
        data=result_vintage[result_vintage['pass_rate'] >= p].reset_index(drop = True),
        linewidth = 0.8
    )
    ax.set_title(title)
    h, l = ax.get_legend_handles_labels() ## https://stackoverflow.com/questions/53734332/seaborn-barplot-legend-labels-lose-color
    ax.legend(h, model_name_list, title = "Model")
    plt.xlabel('Percentile')
    plt.ylabel('Cumulative Bad Debt Rate')
    plt.legend(model_name_list)
    plt.show()
    
def binwise_accumu(
    df,
    unpaid,
    principal,
    title,
    score_list,
    model_name_list,
    p = 0,
    n_bins = 20
):
    '''
    绘制两个图，一个是分箱之后的累积转化率，另一个是按照评分排序之后的逐个样本计算的累积转化率。
    参数解析：
        df: 原来的数据表格。
        unpaid: 说白了就是分子，比如原始label。
        principal: 说白了就是分母，比如一列全都是1的列。请在调用这个函数之前，在df里面加入这一列。
        title: 图的title。
        score_list: 说白了就是预测评分的列名。
        model_name_list: 说白了就是图例里面各列的名字。出于简单可以就直接跟列名一样。
        p: 默认是0，说明这个折线从横轴为0的地方开始画；这个值的取值范围是[0,1)，就是从百分之多少开始绘图。有的时候，折线开头会很高，那么
我们就可以用这个参数跳过前面的部分。
    '''
    result = df
    result.reset_index(drop = True, inplace = True)
    
    bin20_result_concat = []
    breaks_list = []
    for i in range(len(score_list)):
        result.sort_values(by = score_list[i], axis = 0, ascending = True, inplace = True)
        result[f'{score_list[i]}_bin20'] = pd.qcut(result[score_list[i]], n_bins, labels = False) + 1
        
        bins = []
        vintage = []
        for b_i in range(1, n_bins + 1):
            bins.append(b_i)
            v = result[result[f'{score_list[i]}_bin20'] == b_i][unpaid].sum() / result[result[f'{score_list[i]}_bin20'] == b_i][principal].sum()
            vintage.append(v)

        df_i = pd.DataFrame({'bin': bins, 'vintage' : vintage})
        df_i['model'] = str(i + 1)
        bin20_result_concat.append(df_i)
        breaks_list.append(str(i + 1))
        
    bin20_result_concat = pd.concat(bin20_result_concat)
    ##################
    plt.figure(figsize=(10, 5), dpi = 300)
    ax = sns.barplot(
        x="bin", 
        y="vintage", 
        hue="model", 
        data=bin20_result_concat,
        # dodge=False
    )
            
    ax.set_title(title)
    h, l = ax.get_legend_handles_labels() ## https://stackoverflow.com/questions/53734332/seaborn-barplot-legend-labels-lose-color
    ax.legend(h, model_name_list, title = "Model")
    
    first_height_of_patches = [ptc.get_height() for ptc in ax.containers[0]]
    itv = max(first_height_of_patches) / 20
    for i, container in enumerate(ax.containers): 
        ## 某种颜色的所有柱子组成了一个container。
        ## container包含了一系列的patch，每一个patch给我的感觉就是柱状图里的柱那些个矩形。
        xs = [ptc.get_xy()[0] for ptc in container]## get_xy获得每一个柱子的左下角坐标。
        ys = [h + i * itv for h in first_height_of_patches] ## 位置错开一点
        vals = [ptc.get_height() for ptc in container] ## get_height是每一个柱子的高度。
        for x, y, val in zip(xs, ys, vals):
            ax.text(x, y, "%.4f"%val, fontsize = 4) ## 这种方法就是手动在图上标出某个字符串。
        ## ax.bar_label(container, fontsize=3) ## 这种方法也能标注数字，但是标注的位置是定死的，导致标注字符容易重叠。
    
    plt.xlabel("bin")
    plt.ylabel("Bin-Wise Conversion Rate")
    plt.show()       

def load_df_rename_score(fpath, new_colName):
    df = pd.read_csv(fpath).drop_duplicates(subset=None, keep='first', inplace=False).dropna(axis=0,subset = ["trace_id"])
    df.rename(columns = {"score": new_colName}, inplace=True)
    return df


################################################
def score_correlation(
    otherCol, otherScores, 
    info, l, scopes, month_intervals, models_ootStartTime, oot_startDates, observing_timestampe,
    no_model=False,
    score_dir = f"/data/private/public_data/cpu1/minkexiu/DuoTouSubModels/pred_scores_2/",
    time_col = "time_created",
):
    scores = None
    for score_col_name in models_ootStartTime:
        ## 我们的评分:
        # print(score_col_name)
        if scores is None:
            scores = pd.read_csv(os.path.join(score_dir, f"{score_col_name}.csv")).rename(columns={"score": score_col_name})
        else:
            scores = pd.merge(
                scores, pd.read_csv(os.path.join(score_dir, f"{score_col_name}.csv")).rename(columns={"score": score_col_name}), 
                on="trace_id", how="inner"
            )
    scores = pd.merge(scores, otherScores, on="trace_id", how="inner")
    oot_label_score = pd.merge(
        info, scores, on="trace_id", how="inner"
    )
    for mj in month_intervals:
        print(f"******* {mj} *******")
        info_time = oot_label_score[
            (oot_label_score[time_col].astype(str) >= mj[0]) & (oot_label_score[time_col].astype(str) < mj[1])
        ] ### 筛时间

        display(info_time[l+[otherCol]].corr()) ## default correlation method: pearson

def create_cross_table_count_1(df, levelName, colName2, labelColName, n_bins=5):
    '''
    colName2 will be horizontal axis, colName1 will be vertical axis.
    计算一个格子里样本总数。
    '''
    colName1 = f'{levelName}-bin'
    df.loc[:,colName1] = pd.qcut(df[f'{levelName}'], q=n_bins) # pd.cut(df[f'{levelName}'], bins=n_bins) # 
    df.loc[:,f'{colName2}-bin'] = pd.qcut(df[f'{colName2}'], q=n_bins) # pd.cut(df[f'{colName2}'], bins=n_bins) # 
    
    # display(df)
    
    tmp = df.groupby([f'{colName1}', f'{colName2}-bin'])[f"{labelColName}"].count().unstack() 
    
    #######################################################################
    overall1 = df.groupby([f'{colName1}'])[f"{labelColName}"].count()
    tmp = pd.concat(
        [
            pd.DataFrame(
                {f"{levelName}-overall": overall1}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    #######################################################################
    overall2 = df.groupby([f'{colName2}-bin'])[f"{labelColName}"].count()
    overall2.index
    
    tmp = tmp.T
    
    tmp = pd.concat(
        [
            pd.DataFrame(
                {f"{colName2}-overall": overall2}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    tmp = tmp.T
    
    tmp.iloc[-1, -1] = df.shape[0]
    
    return tmp 

def create_cross_table_mean_1(df, levelName, colName2, labelColName, n_bins=5):
    '''
    colName2 will be horizontal axis, colName1 will be vertical axis.
    计算一个格子里样本的下单率。
    '''
    colName1 = f'{levelName}-bin'
    df.loc[:,colName1] = pd.qcut(df[f'{levelName}'], q=n_bins) # pd.cut(df[f'{levelName}'], bins=n_bins) # 
    df.loc[:,f'{colName2}-bin'] = pd.qcut(df[f'{colName2}'], q=n_bins) # pd.cut(df[f'{colName2}'], bins=n_bins) # 
    
    tmp = df.groupby([f'{colName1}', f'{colName2}-bin'])[f"{labelColName}"].mean().unstack() 
    
    #######################################################################
    overall1 = df.groupby([f'{colName1}'])[f"{labelColName}"].mean()
    tmp = pd.concat(
        [
            pd.DataFrame(
                {f"{levelName}-overall": overall1}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    #######################################################################
    overall2 = df.groupby([f'{colName2}-bin'])[f"{labelColName}"].mean()
    overall2.index
    
    tmp = tmp.T
    
    tmp = pd.concat(
        [
            pd.DataFrame(
                {f"{colName2}-overall": overall2}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    tmp = tmp.T
    
    tmp.iloc[-1, -1] = df[labelColName].mean() # df.shape[0]
    
    return tmp 

def create_cross_table_prop_1(df, levelName, colName2, labelColName, n_bins=5):
    '''
    colName2 will be horizontal axis, colName1 will be vertical axis.
    计算一个格子里的样本数占总样本数的比例。
    '''
    colName1 = f'{levelName}-bin'
    df.loc[:,colName1] = pd.qcut(df[f'{levelName}'], q=n_bins) # pd.cut(df[f'{levelName}'], bins=n_bins) # 
    df.loc[:,f'{colName2}-bin'] = pd.qcut(df[f'{colName2}'], q=n_bins) # pd.cut(df[f'{colName2}'], bins=n_bins) # 
    
    tmp = df.groupby([f'{colName1}', f'{colName2}-bin'])[f"{labelColName}"].count().unstack() / df.shape[0]
    
    #######################################################################
    overall1 = df.groupby([f'{colName1}'])[f"{labelColName}"].count() / df.shape[0]
    tmp = pd.concat(
        [
            pd.DataFrame(
                {f"{levelName}-overall": overall1}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    #######################################################################
    overall2 = df.groupby([f'{colName2}-bin'])[f"{labelColName}"].count() / df.shape[0]
    overall2.index
    
    tmp = tmp.T
    
    tmp = pd.concat(
        [
            pd.DataFrame(
                {f"{colName2}-overall": overall2}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    tmp = tmp.T
    
    tmp.iloc[-1, -1] = 1
    
    return tmp 

def draw_level_score_heatmap(oot_label_score, 
                             label = "target_7d", col1 = "level", col2 = "score", 
                             first_subGraph_title = "rate_order",
                             supTitle = "heatmap for data",
                            n_bins = 5,
                             save_img_path = "img.png",
                             myfont = FontProperties(fname='SimHei.ttf'),
                            ):
    cross_7d_mean = create_cross_table_mean_1(
        oot_label_score[[col1, col2, label]], 
        col1, col2, label,
        n_bins=n_bins
    )
    cross_7d_count = create_cross_table_count_1(
        oot_label_score[[col1, col2, label]], 
        col1, col2, label,
        n_bins=n_bins
    )
    cross_7d_prop = create_cross_table_prop_1(
        oot_label_score[[col1, col2, label]], 
        col1, col2, label,
        n_bins=n_bins
    )

    ## 定义一下多图并列。图的大小可能需要调节。
    f, [ax0, ax1, ax2] = plt.subplots(1, 3, figsize=( int(n_bins/5*25), int(n_bins/5*5) ) )
    # f.subplots_adjust(wspace=0.4)
    sns.heatmap(cross_7d_mean * 100, ax = ax0, cmap = sns.cm.rocket_r, annot=True, fmt='.2f')
    sns.heatmap(cross_7d_count.astype(int), ax = ax1, cmap = sns.cm.rocket_r, annot=True, fmt="d")
    sns.heatmap(cross_7d_prop * 100, ax = ax2, cmap = sns.cm.rocket_r, annot=True, fmt='.2f')
    # sns.heatmap(cross_ns_24h * 100, ax = ax3, cmap = sns.cm.rocket_r, annot=True, fmt='.2f')

    titles = [
        first_subGraph_title, 
        "num_sample", 
        "prop_sample",   
    ]

    for i, (aaa, ttl) in enumerate(zip([ax0, ax1, ax2], titles)):
        aaa.set_title(ttl)
        # aaa.set_xticklabels(aaa.get_xticklabels(),rotation=30)
        aaa.set_yticklabels(aaa.get_yticklabels(),rotation=0) # 
        if i == 1:
            continue
        for t in aaa.texts: t.set_text(t.get_text() + " %")

    plt.suptitle(supTitle, fontproperties=myfont)# 
    
    # plt.show()
    
    plt.savefig(
        save_img_path, # os.path.join(save_img_path), 
        bbox_inches='tight'
    )
    
def cross_heatmap(
    otherCol, otherScores, 
    info, l, scopes, labels, month_intervals, models_ootStartTime, oot_startDates, observing_timestampe,
    no_model=False,
    cross_nBins = 10, ## 分箱数量
    score_dir = f"/data/private/public_data/cpu1/minkexiu/DuoTouSubModels/pred_scores_2/",
    save_img_dir = "auc_figs/",
    time_col = "time_created",
    myfont = FontProperties(fname='SimHei.ttf'),
):
    '''
    这个方法主要用在fpd15和max_overdue_days>=15这两种情况。
    '''
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
        
    for score_col_name in models_ootStartTime:
        ## 我们的评分:
        # print(score_col_name)
        score = pd.read_csv(os.path.join(score_dir, f"{score_col_name}.csv"))
        ## 别人的评分: otherScores
        ## 拼起来
        oot_label_score = info.merge(
            score, how = "inner", on = "trace_id",
        ).merge(
            otherScores, how = "inner", on = "trace_id",
        )
        ## 切分：无模型。
        if no_model:
            info_noModel = oot_label_score[oot_label_score["risk_flow_name"].apply(lambda x: '无模型' in f'{x}')] 
        else:
            info_noModel = oot_label_score 
            
        ## 切表现期：
        info_limited = info_noModel[(observing_timestampe - info_noModel["index1_billing_timestamp"])/1000/3600/24 > 15]

        for mj in month_intervals:
            info_time = info_limited[
                (info_limited[time_col].astype(str) >= mj[0]) & (info_limited[time_col].astype(str) < mj[1])
            ] ### 筛时间
            ## 这里我们就不分scope了。以后要切分再加吧。
            for i_, scope in enumerate(scopes):
                if scope != "ALL": ## 暂且不分scope，遇到scope为不是ALL的就跳过。如果以后要划分scope，可以在这里加上切分的代码。
                    continue
                print(f"******* {score_col_name} {mj} #Sample: {info_time.shape[0]}*******") ## 不显示 {scope} 

                for label in labels:
                    draw_level_score_heatmap(
                        info_time[[otherCol, "score", label]], 
                        label = label, col1 = otherCol, col2 = "score",
                        supTitle = f"""vert coord: {otherCol}; hori coord: {score_col_name} score; {mj[0].replace('-', '_')} to {mj[1].replace('-', '_')}; label: {label}""", 
                        first_subGraph_title = f"rate_{label}",
                        n_bins = cross_nBins,
                        save_img_path= os.path.join(
                            save_img_dir, 
                            f"{otherCol}-{score_col_name}-{mj[0].replace('-', '_')}_{mj[1].replace('-', '_')}-{scope}-{label}.png"
                        ),
                        myfont = myfont
                    )

def cross_heatmap_multiplePDs(
    otherCol, otherScores, 
    info, l, scopes, labels, month_intervals, models_ootStartTime, oot_startDates, observing_timestampe,
    no_model=False,
    cross_nBins = 10, ## 分箱数量
    score_dir = f"/data/private/public_data/cpu1/minkexiu/DuoTouSubModels/pred_scores_2/",
    save_img_dir = "auc_figs/",
    time_col = "time_created",
    myfont = FontProperties(fname='SimHei.ttf'),
):
    '''
    这个方法主要用在若干种xPDy这种指标，比如1pd15也就是fpd15这种。
    '''
    for label in labels:
        if not re.match("\d+pd\d+", label):
            raise Exception("All labels should be like `\d+pd\d+` e.g., 1pd15, 2pd30")
    
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
        
    for score_col_name in models_ootStartTime:
        ## 我们的评分:
        # print(score_col_name)
        score = pd.read_csv(os.path.join(score_dir, f"{score_col_name}.csv"))
        ## 别人的评分: otherScores
        ## 拼起来
        oot_label_score = info.merge(
            score, how = "inner", on = "trace_id",
        ).merge(
            otherScores, how = "inner", on = "trace_id",
        )
        ## 切分：无模型。
        if no_model:
            info_noModel = oot_label_score[oot_label_score["risk_flow_name"].apply(lambda x: '无模型' in f'{x}')] 
        else:
            info_noModel = oot_label_score 
        # info_limited = info_noModel # info_noModel[(observing_timestampe - info_noModel["index1_billing_timestamp"])/1000/3600/24 > 15]

        for mj in month_intervals:
            info_time = info_noModel[
                (info_noModel[time_col].astype(str) >= mj[0]) & (info_noModel[time_col].astype(str) < mj[1])
            ] ### 筛时间
            ## 这里我们就不分scope了。以后要切分再加吧。
            for i_, scope in enumerate(scopes):
                if scope != "ALL": ## 暂且不分scope，遇到scope为不是ALL的就跳过。如果以后要划分scope，可以在这里加上切分的代码。
                    continue  
                print(f"******* {score_col_name} {mj}*******") ## 不显示 {scope} # print(f"******* {score_col_name} {mj} #Sample: {info_time.shape[0]}*******") ## 不显示 {scope} 
                for label in labels:
                    _1, _2 = label.split("pd")
                    info_limited = info_time[(observing_timestampe - info_time[f"index{_1}_billing_timestamp"])/1000/3600/24 > int(_2)]
                    if info_limited.shape[0] <= 0:
                        print(label, info_limited.shape, "no data, skipping...")
                        continue
                    # print(info_limited.shape)
                    draw_level_score_heatmap(
                        info_limited[[otherCol, "score", label]], 
                        label = label, col1 = otherCol, col2 = "score",
                        supTitle = f"""vert coord: {otherCol}; hori coord: {score_col_name} score; {mj[0].replace('-', '_')} to {mj[1].replace('-', '_')}; label: {label}""", 
                        first_subGraph_title = f"rate_{label}",
                        n_bins = cross_nBins,
                        save_img_path= os.path.join(
                            save_img_dir, 
                            f"{otherCol}-{score_col_name}-{mj[0].replace('-', '_')}_{mj[1].replace('-', '_')}-{scope}-{label}.png"
                        ),
                        myfont = myfont
                    )
######################################################################################                
def total_auc(
    labels,
    info,
    observing_timestampe, 
    tmp_list, 
    scopes, 
    no_model = False, 
    save_path = "total_auc.csv", 
    score_dir = "/data/private/public_data/cpu1/minkexiu/DuoTouSubModels/pred_scores_2/"
):
#     for scope in scopes:
    perf_comparison = pd.DataFrame({"idx": labels})
    #############################################################
#         ## 切分scope。
#         if scope == "RNG":
#             info_scope = info[info.scope.isin(["R", "N", "G"])]
#         elif scope in ["R", "N", "G", "L"]:
#             info_scope = info[info.scope == scope]
#         elif scope == "L/R":
#             info_scope = info[(info.scope == 'L') & (info.loan_times >= 1) & (info.balance_orders == 0)]
#         elif scope == "L/N":
#             info_scope = info[(info.scope == 'L') & (info.loan_times >= 2) & (info.balance_orders >= 1)]
#         elif scope == "L/G":
#             info_scope = info[(info.scope == 'L') & (info.loan_times == 1) & (info.balance_orders >= 1)]
#         elif scope == "(L/R,R)":
#             info_scope = info[
#                 (info.scope == "R") | ((info.scope == 'L') & (info.loan_times >= 1) & (info.balance_orders == 0))
#             ]
#         elif scope == "(L/N,N)":
#             info_scope = info[
#                 (info.scope == "N") | ((info.scope == 'L') & (info.loan_times >= 2) & (info.balance_orders >= 1))
#             ]
#         elif scope == "(L/G,G)":
#             info_scope = info[
#                 (info.scope == "G") | ((info.scope == 'L') & (info.loan_times == 1) & (info.balance_orders >= 1))
#             ]

    if no_model:
        info_scope = info[info["risk_flow_name"].apply(lambda x: '无模型' in f'{x}')] ## 如果是无模型，就加上这个筛选条件，否则就去掉。
    else:
        info_scope = info # [info["risk_flow_name"].apply(lambda x: '无模型' in f'{x}')] ## 如果是无模型，就加上这个筛选条件，否则就去掉。
    
    ## 这个是一个一个的列。
    for score_col_name in tmp_list:
        scores = pd.read_csv(os.path.join(score_dir, f"{score_col_name}.csv"))
        # print("  ", score_col_name)
        col_vals = []
        n_samples = []
        pos_prop = []
        ## 这些是一个一个行。
        for label in labels:
            print(f"{score_col_name}, {label}")
#             oot_end = calc_oot_limit_with_specificTime(
#                 label.split("_")[-1], observing_timestampe
#             )
#             info_limited = info_scope[(pd.to_datetime(info_scope.trace_time, unit = "ms") + timedelta(hours=8)) <= oot_end]

            info_limited = info_scope[(observing_timestampe - info_scope["index1_billing_timestamp"])/1000/3600/24 > 15]
            #############################


            if info_limited.shape[0] == 0:
                auc = 0 ## 如果没有数据，那给一个默认值就是0
            else:
                info_limited = info_limited.merge(scores, how='inner', left_on='trace_id', right_on='trace_id')
                try:
                    auc = roc_auc_score(
                        info_limited[label], 
                        info_limited.loc[:, "score"]
                    )
                except Exception as e:
                    if "Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required." in str(e):
                        auc = -0.01
                    elif "Only one class present in y_true. ROC AUC score is not defined in that case." in str(e):
                        auc = -0.02
                    else:
                        print(str(e))
                        auc = -0.03
            n_samples.append(info_limited.shape[0])
            pos_prop.append(info_limited[label].mean())
            col_vals.append(auc)
        if "n_samples" not in perf_comparison.columns:
            perf_comparison["n_samples"] = n_samples
        if "pos_prop" not in perf_comparison.columns:
            perf_comparison["pos_prop"] = pos_prop
        perf_comparison[score_col_name] = col_vals
        # display(perf_comparison)

    # print("="*30, scope, scope_to_chinese(scope), "="*30)
    
    perf_comparison.to_csv(save_path, index=False)
    
    perf_comparison = perf_comparison.style.background_gradient(
        cmap="RdYlGn", # cm
        subset=perf_comparison.columns[3:], 
        axis=1
    )
    display(perf_comparison)
    
    
def do_something(observing_timestampe, info, month_intervals, oot_startDates, models_ootStartTime, labels, scopes, score_col_name, no_model = False, score_dir = f"/data/private/public_data/cpu1/minkexiu/DuoTouSubModels/pred_scores_2/"):
    scores = pd.read_csv(os.path.join(score_dir, f"{score_col_name}.csv"))
    
    idx = []
    for label in labels:
        for scope in scopes:
            idx.append(f"{label}-{scope}")
    perf_comparison = pd.DataFrame({"idx": idx + ["model", "order"]}, index=None)
    num_count = pd.DataFrame({"idx": idx + ["model", "order"]}, index=None)
    pos_prop = pd.DataFrame({"idx": idx + ["model", "order"]}, index=None)
    
    oot_startDate = models_ootStartTime[score_col_name]
    idx_ = oot_startDates.index(oot_startDate)
    col_count = 0
    for j in tqdm.tqdm(range(0, len(month_intervals))): ### 选择预测的月份
        if j < idx_: ## 遍历所有的月份，但并非所有的月份都会运行下面的代码。
            continue
        mj = month_intervals[j]
        # print(" ", mj)
        info_time = info[
            (info.time_created.astype(str) >= mj[0]) & (info.time_created.astype(str) < mj[1])
        ] ### 筛时间
#         print("info_time.shape", info_time.shape)
        
        auc_dict = {}
        num_dict = {}
        prop_dict = {}
        for scope in scopes:

            ## 切分scope。
            if scope == "ALL":
                info_limited = info_time
            elif scope == "RNG":
                info_limited = info_time[info_time.trace_id_scope.isin(["R", "N", "G"])]
            #############################################################
            elif scope in ["R", "N", "G", "L"]:
                info_limited = info_time[info_time.trace_id_scope == scope]
            elif scope == "L/R":
                info_limited = info_time[(info_time.trace_id_scope == 'L') & (info_time.loan_times >= 1) & (info_time.balance_orders == 0)]
            elif scope == "L/N":
                info_limited = info_time[(info_time.trace_id_scope == 'L') & (info_time.loan_times >= 2) & (info_time.balance_orders >= 1)]
            elif scope == "L/G":
                info_limited = info_time[(info_time.trace_id_scope == 'L') & (info_time.loan_times == 1) & (info_time.balance_orders >= 1)]

            #############################################################
            elif scope == "(L/R,R)":
                info_limited = info_time[
                    (info_time.trace_id_scope == "R") | ((info_time.trace_id_scope == 'L') & (info_time.loan_times >= 1) & (info_time.balance_orders == 0))
                ]
            elif scope == "(L/N,N)":
                info_limited = info_time[
                    (info_time.trace_id_scope == "N") | ((info_time.trace_id_scope == 'L') & (info_time.loan_times >= 2) & (info_time.balance_orders >= 1))
                ]
            elif scope == "(L/G,G)":
                info_limited = info_time[
                    (info_time.trace_id_scope == "G") | ((info_time.trace_id_scope == 'L') & (info_time.loan_times == 1) & (info_time.balance_orders >= 1))
                ]
                
            # print("info_limited.shape", info_limited.shape)
            part_data = info_limited.merge(scores, how='inner', left_on='trace_id', right_on='trace_id') ### 附上评分
            # print(part_data.shape)
            
            for label in labels:

                ## 新增的部分：满足观测期
#                 oot_end = calc_oot_limit_with_specificTime(
#                     label.split("_")[-1], observing_timestampe
#                 ) 
                    
                ## 满足观测期
                part_data_limited = part_data[(observing_timestampe - part_data["index1_billing_timestamp"])/1000/3600/24 > 15]
                # print(part_data_limited.shape)
                
                if no_model:
                    part_data_oot = part_data_limited[part_data_limited["risk_flow_name"].apply(lambda x: '无模型' in f'{x}')] ## 这个是筛选无模型用的。
                else:
                    part_data_oot = part_data_limited # [part_data_limited["risk_flow_name"].apply(lambda x: '无模型' in f'{x}')] ## 这个是筛选无模型用的。
                # print(part_data_oot.shape)
    # [(pd.to_datetime(part_data.trace_time, unit = "ms") + timedelta(hours=8)) <= oot_end]

                try:
                    auc = roc_auc_score(
                        part_data_oot[label], 
                        ## 这个地方比较拧巴，因为形如“july_only”的分数csv文件里面的列名形如“jul”
                        part_data_oot.loc[:, "score"] # score_col_name if "_only" not in score_col_name else score_col_name.split("_")[0] 
                    )
                except Exception as e:
                    if "Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required." in str(e):
                        auc = -0.01
                    elif "Only one class present in y_true. ROC AUC score is not defined in that case." in str(e):
                        auc = -0.02
                    else:
                        auc = -0.03
                auc_dict[f"{label}-{scope}"] = auc
                num_dict[f"{label}-{scope}"] = part_data_oot.shape[0]
                prop_dict[f"{label}-{scope}"] = part_data_oot[label].mean()
        ##################################################
        col_count += 1
        perf_comparison[mj[0]] = np.array([auc_dict[i] for i in idx] + [score_col_name, col_count])
        num_count[mj[0]] = np.array([num_dict[i] for i in idx] + [score_col_name, col_count])
        pos_prop[mj[0]] = np.array([prop_dict[i] for i in idx] + [score_col_name, col_count])
    
    final_df = perf_comparison.set_index("idx")
    final_df.to_csv(f"auc_rsts/{score_col_name}_auc.csv")
    
    final_df1 = num_count.set_index("idx")
    final_df1.to_csv(f"auc_rsts/{score_col_name}_nSample.csv")
    
    final_df2 = pos_prop.set_index("idx")
    final_df2.to_csv(f"auc_rsts/{score_col_name}_posProp.csv")
    return final_df, final_df1, final_df2


def monthwise_perf(
    l, observing_timestampe, info, month_intervals, oot_startDates, models_ootStartTime, labels, scopes,
    no_model=False, save_name = "monthwise_perf", 
    save_dir = "auc_tabs", score_dir = f"/data/private/public_data/cpu1/minkexiu/DuoTouSubModels/pred_scores_2/"
):
    dfs, df_cnt, df_prop = [], [], []
    for model_n in list(models_ootStartTime.keys()):
        print(model_n)
        _1, _2, _3 = do_something(observing_timestampe, info, month_intervals, oot_startDates, models_ootStartTime, labels, scopes, model_n, no_model, score_dir)
        
#         display(_1)
#         display(_2)
#         display(_3)
        
        dfs.append(_1)
        df_cnt.append(_2)
        df_prop.append(_3)
    ## 其实df_cnt里面所有的表格，都应该是一样的。df_prop内部亦然。
    ## 所以任意取其中一个就行了。
    
    ## 把要画图的都画在一起
    df_new = pd.concat(dfs, axis=1)
    df_T = df_new.T
    df_T.reset_index(inplace=True)
    
    save_fig_dir = "auc_figs"
    ## 这个决定了要画多少个图
    label_scopes = []
    for label in labels:
        for scope in scopes:
            label_scopes.append(f"{label}-{scope}") ## .replace('/', '_')
    # label_scopes
    
    cols_to_be_seen = label_scopes # ["fpd15-ALL"] ## 相当于从label_scopes里面找一部分来看。

    new_df_T = df_T[["index", "model", "order"] + cols_to_be_seen].sort_values(["index", "order"])
    length = len(month_intervals)

    for col_2bseen in cols_to_be_seen:
        part = None
        for i, mi in enumerate(month_intervals):    
            part_ = new_df_T.iloc[i * len(l) :(i+1) * len(l), :]
            index_of_part_ = part_["index"].to_list()[0]
            part_1 = pd.DataFrame(
                {
                    "model": part_["model"],
                    index_of_part_: part_[col_2bseen].astype(float)
                }
            )
            if part is None:
                part = part_1
            else:
                part = pd.merge(part, part_1, how="inner", on="model")


        part = part.set_index("model").T

        part = pd.concat([part, df_prop[0].T[col_2bseen]], axis=1) ## 连上正样本比例。
        part[col_2bseen] = part[col_2bseen].astype(float) 
        part.rename(columns={col_2bseen: "posProp"}, inplace=True)
        part = pd.concat([part, df_cnt[0].T[col_2bseen]], axis=1) ## 连上样本总数。
        part.rename(columns={col_2bseen: "#sample"}, inplace=True)
        part = part[part.columns[::-1]] ## 列排序给它反序一下。

        if "-L" in col_2bseen: ## 这里我们把其他的scope的过滤掉。
            continue

        part.to_csv(
            os.path.join(save_dir, f"{save_name}-{col_2bseen}.csv"), 
            # index=False
        ) 
        print("=" * 30, col_2bseen, "=" * 30)    
        part_colorful = part.style.background_gradient(
            cmap="RdYlGn", # cm
            subset=part.columns[2:], 
            axis=1
        )
        display(part_colorful)

        
    model_list = list(models_ootStartTime.keys()) ## 这个决定了一个图里有多少子图
    for label in labels:

        f, axes = plt.subplots(1, len(scopes), figsize=(20, 5), dpi=500)

        for i, scope in enumerate(scopes):

            label_scope = f"{label}-{scope}"

            print(label_scope)

            new_x = []
            for idx_, cnt_, prop_ in zip(
                df_T["index"],
                df_cnt[0].loc[label_scope, :].to_list() * len(model_list), 
                df_prop[0].loc[label_scope, :].to_list() * len(model_list), 
            ):
                new_x.append(f"{idx_}\n#sample: {cnt_}\nposProp: {float(format(float(prop_), '.2g'))}") 

            df_plot = pd.DataFrame(
                {
                    "time": new_x, # list(df_T["index"]),
                    "model": list(df_T["model"]), 
                    "auc": list(df_T[label_scope])
                }
            )
            df_plot["auc"] = df_plot["auc"].astype(float)
            ########################
            axes[i].set_title(f"Scope: {scope}")# translate_labelScope(label_scope).split(",")[1]
            sns.lineplot(x="time", y="auc", hue="model", data=df_plot, ax=axes[i]) # 
            axes[i].tick_params(axis='x', labelrotation=90)

        plt.suptitle("OOT label: " + label) # translate_labelScope(label_scope).split(",")[0]
        plt.savefig(
            os.path.join(save_fig_dir, label + '-comb.png'), 
            bbox_inches='tight'
        )
        
def bin_bad_rate(
    info, l, scopes, labels, month_intervals, models_ootStartTime, oot_startDates, observing_timestampe,
    no_model=False,
    convert_nBins = 10, ## 分箱数量
    score_dir = f"/data/private/public_data/cpu1/minkexiu/DuoTouSubModels/pred_scores_2/"
):
    '''
    bin的策略是等频分箱。
    '''
    
    format_dict = {colName: '{:.2%}' for colName in models_ootStartTime}
    color = ["lightblue", "lightgreen"]

    scores_all = {}
    for score_col_name in models_ootStartTime:
        scores_all[score_col_name] = pd.read_csv(os.path.join(score_dir, f"{score_col_name}.csv"))

    if no_model:
        info_noModel = info[info["risk_flow_name"].apply(lambda x: '无模型' in f'{x}')] ## 这个是筛选无模型用的。
    else:
        info_noModel = info
        
        
    for label in labels:
        print("*" * 30, label, "*" * 30, )
        for mj in month_intervals:
            info_time = info_noModel[
                (info_noModel.time_created.astype(str) >= mj[0]) & (info_noModel.time_created.astype(str) < mj[1])
            ] ### 筛时间
            for i_, scope in enumerate(scopes):
                if scope == "L": ## 暂且不分scope
                    continue
                print(mj, scope)
                info_limited = info_time[(observing_timestampe - info_time["index1_billing_timestamp"])/1000/3600/24 > 15]
                cross_ = {}
                ### 搜集评分
                for score_col_name in models_ootStartTime:
                    scores = scores_all[score_col_name] # pd.read_csv(os.path.join(score_dir, f"{score_col_name}.csv"))
                    part_data = info_limited.merge(scores, how='inner', left_on='trace_id', right_on='trace_id')

                    if label not in cross_:
                        cross_[label] = part_data[label].reset_index(drop=True)
                        print("#Sample: {}; posRate: {}".format(part_data.shape[0], part_data[label].mean()))

                    cross_[score_col_name] = part_data.loc[:, "score"].reset_index(drop=True)
                #######################################
                rst = {}
                cross = pd.DataFrame(cross_)

                for colName in models_ootStartTime:
                    bin_col_name = f"{colName}-bin"
                    cross.loc[:, bin_col_name], short_score_bin = pd.qcut(
                        cross[colName], q=convert_nBins, retbins=True, labels=False
                    )
                    group_new = cross.groupby(bin_col_name)
                    v1_bin = group_new[label].sum()/group_new[label].count()
                    rst[colName] = v1_bin

                display(
                    (
                        pd.DataFrame(rst)
                        .style
                        .format(format_dict)
                        .bar(
                            color=color[i_ % len(color)],#color[scope], 
                            align='zero'
                        )
                    )
                )
                
def calc_oot_limit_with_specificTime(label_interval, specific_time):
    '''
    info: 包含trace_time（以毫秒为单位的时间戳）的数据集
    label_interval: {'hour', "1d", "3d", "7d", "15d", "35d"}.
    specific_time: unix时间戳，毫秒为单位
    '''
    assert label_interval in ['hour', "1d", "3d", "7d", "14d", "15d", "35d"], '''the 'label_interval' must be in ['hour', "1d", "3d", "7d", "14d", "15d", "35d"] '''
    if label_interval == "hour":
        oot_end = pd.to_datetime(specific_time, unit = "ms") + timedelta(hours=8) - timedelta(hours=1)
    else:
        num_days = int(label_interval[0:-1])
        oot_end = pd.to_datetime(specific_time, unit = "ms") + timedelta(hours=8) - timedelta(days=num_days)
    return oot_end

def create_cross_table_count_2(df, levelName, colName2, labelColName, n_bins=5):
    '''
    colName2 will be horizontal axis, colName1 will be vertical axis.
    计算一个格子里样本总数。
    '''
    colName1 = f'{levelName}-bin'
    df.loc[:,colName1] = pd.qcut(df[f'{levelName}'], q=n_bins) 
    df.loc[:,f'{colName2}-bin'] = df.loc[:,f'{colName2}'] 
    
    # display(df)
    
    tmp = df.groupby([f'{colName1}', f'{colName2}-bin'])[f"{labelColName}"].count().unstack() 
    
    #######################################################################
    overall1 = df.groupby([f'{colName1}'])[f"{labelColName}"].count()
    tmp = pd.concat(
        [
            pd.DataFrame(
                {f"{levelName}-overall": overall1}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    #######################################################################
    overall2 = df.groupby([f'{colName2}-bin'])[f"{labelColName}"].count()
    overall2.index
    
    tmp = tmp.T
    
    tmp = pd.concat(
        [
            pd.DataFrame(
                {f"{colName2}-overall": overall2}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    tmp = tmp.T
    
    tmp.iloc[-1, -1] = df.shape[0]
    
    return tmp 

def create_cross_table_mean_2(df, levelName, colName2, labelColName, n_bins=5):
    '''
    colName2 will be horizontal axis, colName1 will be vertical axis.
    计算一个格子里样本的下单率。
    '''
    colName1 = f'{levelName}-bin'
    df.loc[:,colName1] = pd.qcut(df[f'{levelName}'], q=n_bins) 
    df.loc[:,f'{colName2}-bin'] = df.loc[:,f'{colName2}'] 
    
    tmp = df.groupby([f'{colName1}', f'{colName2}-bin'])[f"{labelColName}"].mean().unstack() 
    
    #######################################################################
    overall1 = df.groupby([f'{colName1}'])[f"{labelColName}"].mean()
    tmp = pd.concat(
        [
            pd.DataFrame(
                {f"{levelName}-overall": overall1}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    #######################################################################
    overall2 = df.groupby([f'{colName2}-bin'])[f"{labelColName}"].mean()
    overall2.index
    
    tmp = tmp.T
    
    tmp = pd.concat(
        [
            pd.DataFrame(
                {f"{colName2}-overall": overall2}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    tmp = tmp.T
    
    tmp.iloc[-1, -1] = df[labelColName].mean() # df.shape[0]
    
    return tmp 

def create_cross_table_sum_2(df, levelName, colName2, labelColName, n_bins=5):
    '''
    colName2 will be horizontal axis, colName1 will be vertical axis.
    计算一个格子里样本的下单率。
    '''
    colName1 = f'{levelName}-bin'
    df.loc[:,colName1] = pd.qcut(df[f'{levelName}'], q=n_bins) 
    df.loc[:,f'{colName2}-bin'] = df.loc[:,f'{colName2}'] 
    
    tmp = df.groupby([f'{colName1}', f'{colName2}-bin'])[f"{labelColName}"].sum().unstack() 
    
    #######################################################################
    overall1 = df.groupby([f'{colName1}'])[f"{labelColName}"].sum()
    tmp = pd.concat(
        [
            pd.DataFrame(
                {f"{levelName}-overall": overall1}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    #######################################################################
    overall2 = df.groupby([f'{colName2}-bin'])[f"{labelColName}"].sum()
    overall2.index
    
    tmp = tmp.T
    
    tmp = pd.concat(
        [
            pd.DataFrame(
                {f"{colName2}-overall": overall2}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    tmp = tmp.T
    
    tmp.iloc[-1, -1] = df[labelColName].sum() # df.shape[0]
    
    return tmp 

def create_cross_table_prop_2(df, levelName, colName2, labelColName, n_bins=5):
    '''
    colName2 will be horizontal axis, colName1 will be vertical axis.
    计算一个格子里的样本数占总样本数的比例。
    '''
    colName1 = f'{levelName}-bin'
    df.loc[:,colName1] = pd.qcut(df[f'{levelName}'], q=n_bins) 
    df.loc[:,f'{colName2}-bin'] = df.loc[:,f'{colName2}'] 
    
    tmp = df.groupby([f'{colName1}', f'{colName2}-bin'])[f"{labelColName}"].count().unstack() / df.shape[0]
    
    #######################################################################
    overall1 = df.groupby([f'{colName1}'])[f"{labelColName}"].count() / df.shape[0]
    tmp = pd.concat(
        [
            pd.DataFrame(
                {f"{levelName}-overall": overall1}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    #######################################################################
    overall2 = df.groupby([f'{colName2}-bin'])[f"{labelColName}"].count() / df.shape[0]
    overall2.index
    
    tmp = tmp.T
    
    tmp = pd.concat(
        [
            pd.DataFrame(
                {f"{colName2}-overall": overall2}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    tmp = tmp.T
    
    tmp.iloc[-1, -1] = 1
    
    return tmp 


def draw_level_score_heatmap_2(
    oot_label_score, labels, 
    col1 = "level", 
    col2 = "score", 
    first_subGraph_title = "rate_order",
    supTitle = "heatmap for data",
    n_bins = 5,
    save_img_path = "img.png",
    myfont = FontProperties(fname='SimHei.ttf'),
    figSize = (10, 10)
):
    
    oot_label_score["dummy"] = 1
    ## 定义一下多图并列。图的大小可能需要调节。
    f, axes = plt.subplots(1, 2 + len(labels) * 2, figsize=figSize )
    # f.subplots_adjust(wspace=0.4)
    
    cross_7d_count = create_cross_table_count_2(
        oot_label_score[[col1, col2, "dummy"]], 
        col1, col2, "dummy",
        n_bins=n_bins
    )
    cross_7d_prop = create_cross_table_prop_2(
        oot_label_score[[col1, col2, "dummy"]], 
        col1, col2, "dummy",
        n_bins=n_bins
    )
    sns.heatmap(cross_7d_count.astype(int), ax = axes[0], cmap = sns.cm.rocket_r, annot=True, fmt="d")
    sns.heatmap(cross_7d_prop * 100, ax = axes[1], cmap = sns.cm.rocket_r, annot=True, fmt='.2f', yticklabels=False)
    
    
    for i, label in enumerate(labels):
        cross_7d_mean = create_cross_table_mean_2(
            oot_label_score[[col1, col2, label]], 
            col1, col2, label,
            n_bins=n_bins
        )
        sns.heatmap(cross_7d_mean * 100, ax = axes[2+i*2], cmap = sns.cm.rocket_r, annot=True, fmt='.2f', yticklabels=False)
        sns.heatmap(cross_7d_mean / cross_7d_mean.iloc[-1, -1], ax = axes[3+i*2], cmap = sns.cm.rocket_r, annot=True, fmt='.2f', yticklabels=False)
    
    # sns.heatmap(cross_ns_24h * 100, ax = ax3, cmap = sns.cm.rocket_r, annot=True, fmt='.2f')

    titles = [
        "sample number", 
        "sample proportion",   
    ]
    for lb in labels:
        titles += [f"posRate-{lb}", f"lift_on_{col1}_{col2}_ovrl-{lb}"]

    for i, (aaa, ttl) in enumerate(zip(axes, titles)):
        aaa.set_title(ttl)
        # aaa.set_xticklabels(aaa.get_xticklabels(),rotation=30)
        aaa.set_yticklabels(aaa.get_yticklabels(),rotation=0) # 
        if (i == 0) or ("lift" in ttl):
            continue
        for t in aaa.texts: t.set_text(t.get_text() + " %")

    plt.suptitle(supTitle, fontproperties=myfont)# 
    
    # plt.show()
    
    plt.savefig(
        save_img_path, # os.path.join(save_img_path), 
        bbox_inches='tight'
    )

def cross_heatmap_multiplePDs_2(
    days, scheme_type, model_name, dt, nds, labels,
    otherCol, 
    # otherScores, 
    info, month_intervals, # models_ootStartTime,
    scopes = ["ALL"],
    no_model=False,
    cross_nBins = 10, ## 分箱数量
    score_file_path_format="/data/private/public_data/cpu1/minkexiu/Liucun/trained_models/lbl_{}d/bhv/dropSchemeV4_noXun_bhvAndShoudai/feasSelecting/{}/oot_{}.csv",
    save_img_dir = "cross_test_dir/",
    time_col = "time_created",
    myfont = FontProperties(fname='SimHei.ttf'),
):
    '''
    时间的分割变成了用户区分。
    '''
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    score_col_name = model_name
    ## 我们的评分:
    # print(score_col_name)
    score = pd.read_csv(
        score_file_path_format.format(days, scheme_type, score_col_name, dt[-4:])
    )
    ## 别人的评分: otherScores
    ## 拼起来
    # otherScores.trace_id = otherScores.trace_id.astype(str) + f"-{dt[-4:]}"
    
    oot_label_score = info.merge(
        score, how = "inner", left_on = "loan_account_id", right_on = "trace_id",
    )
    ## 切分：无模型。
    if no_model:
        info_noModel = oot_label_score[oot_label_score["risk_flow_name"].apply(lambda x: '无模型' in f'{x}')] 
    else:
        info_noModel = oot_label_score 
        
    for mj in month_intervals:
        info_time = info_noModel[
            (info_noModel[time_col] >= mj[0]) & (info_noModel[time_col] < mj[1])
        ] ### 筛时间
        ## 这里我们就不分scope了。以后要切分再加吧。
        for i_, scope in enumerate(scopes):
            if scope != "ALL": ## 暂且不分scope，遇到scope为不是ALL的就跳过。如果以后要划分scope，可以在这里加上切分的代码。
                continue  
            print(f"******* {score_col_name} {mj}*******") ## 不显示 {scope} # print(f"******* {score_col_name} {mj} #Sample: {info_time.shape[0]}*******") ## 不显示 {scope} 
            info_limited = info_time # [(observing_timestampe - info_time[f"index{_1}_billing_timestamp"])/1000/3600/24 > int(_2)]

#             if info_limited.shape[0] <= 0:
#                 print(label, info_limited.shape, "no data, skipping...")
#                 continue

            if not os.path.exists(save_img_dir):
                os.makedirs(save_img_dir)
                # print(info_limited.shape)
                
            act_lbls = []
            for lb in labels:
                if lb in info_limited.columns:
                    act_lbls.append(lb)
            if len(act_lbls) > 0:
                draw_level_score_heatmap_2(
                    info_limited[[otherCol, "score"] + act_lbls], act_lbls,
                    col1 = "score", col2 = otherCol,
                    supTitle = f"""vert coord: 模型分; hori coord: 撞库规则; 结清距今 {mj[0]} 到 {mj[1]} 天的用户; OOT date: {dt}""", 
                    n_bins = cross_nBins,
                    save_img_path= os.path.join(
                        save_img_dir, 
                        f"{dt}-({mj[0]}_{mj[1]}]-[onlineModel]x[rule]-{scope}.png"
                    ),    
                    myfont = myfont
                )
            else:
                print("not enough label")
                
                
def draw_level_score_heatmap_3(
    oot_label_score, labels, 
    col1 = "level", 
    col2 = "score", 
    first_subGraph_title = "rate_order",
    supTitle = "heatmap for data",
    n_bins = 5,
    save_img_path = "img.png",
    myfont = FontProperties(fname='SimHei.ttf'),
    figSize = (10, 10)
):
    
    oot_label_score["dummy"] = 1
    ## 定义一下多图并列。图的大小可能需要调节。
    f, axes = plt.subplots(1, 2 + len(labels) * 4, figsize=figSize )
    # f.subplots_adjust(wspace=0.4)
    
    cross_7d_count = create_cross_table_count_2(
        oot_label_score[[col1, col2, "dummy"]], 
        col1, col2, "dummy",
        n_bins=n_bins
    )
    cross_7d_prop = create_cross_table_prop_2(
        oot_label_score[[col1, col2, "dummy"]], 
        col1, col2, "dummy",
        n_bins=n_bins
    )
    sns.heatmap(cross_7d_count.astype(int), ax = axes[0], cmap = sns.cm.rocket_r, annot=True, fmt="d")
    sns.heatmap(cross_7d_prop * 100, ax = axes[1], cmap = sns.cm.rocket_r, annot=True, fmt='.2f', yticklabels=False)
    
    # col1
    
    for i, label in enumerate(labels):
        cross_7d_mean = create_cross_table_mean_2(
            oot_label_score[[col1, col2, label]], 
            col1, col2, label,
            n_bins=n_bins
        )
        sns.heatmap(cross_7d_mean * 100, ax = axes[2+i*2], cmap = sns.cm.rocket_r, annot=True, fmt='.2f', yticklabels=False)
        sns.heatmap(cross_7d_mean / cross_7d_mean.iloc[-1, -1], ax = axes[3+i*2], cmap = sns.cm.rocket_r, annot=True, fmt='.2f', yticklabels=False)
    
        
        ################################ col1是分数。      
        new_df = [([None,] * cross_7d_count.shape[1]) for i in range(cross_7d_count.shape[0])]
        oot_label_score_1 = oot_label_score.copy(deep=True)
        oot_label_score_1 = oot_label_score_1.sort_values([col1]).reset_index(drop=True)
        ## 核心部分
        total_num = 0
        for h in range(cross_7d_count.shape[0] - 1):
            for j in range(cross_7d_count.shape[1] - 1):
                cur_num = cross_7d_count.iloc[h, j]
                new_df[h][j] = oot_label_score_1.loc[total_num: total_num + cur_num - 1, label].mean()
                total_num += cur_num
        ## 底下一行
        total_num = 0
        for j in range(cross_7d_count.shape[1] - 1):
            cur_num = cross_7d_count.iloc[-1, j]
            new_df[cross_7d_count.shape[0] - 1][j] = oot_label_score_1.loc[total_num: total_num + cur_num - 1, label].mean()
            total_num += cur_num
        ## 右边一列
        total_num = 0
        for h in range(cross_7d_count.shape[0] - 1):
            cur_num = cross_7d_count.iloc[h, -1]
            new_df[h][cross_7d_count.shape[1] - 1] = oot_label_score_1.loc[total_num: total_num + cur_num - 1, label].mean()
            total_num += cur_num
        ## 右下角
        total_num = 0
        cur_num = cross_7d_count.iloc[-1, -1]
        new_df[-1][-1] = oot_label_score_1.loc[total_num: total_num + cur_num - 1, label].mean()
        total_num += cur_num
        
        cross_7d_mean_col1 = pd.DataFrame(new_df, columns = cross_7d_count.columns, index=cross_7d_count.index)
        
        sns.heatmap(cross_7d_mean_col1 * 100, ax = axes[4+i*4], cmap = sns.cm.rocket_r, annot=True, fmt='.2f', yticklabels=False)
        sns.heatmap(cross_7d_mean_col1 / cross_7d_mean_col1.iloc[-1, -1], ax = axes[5+i*4], cmap = sns.cm.rocket_r, annot=True, fmt='.2f', yticklabels=False)
        ################################
    
    # sns.heatmap(cross_ns_24h * 100, ax = ax3, cmap = sns.cm.rocket_r, annot=True, fmt='.2f')
    titles = [
        "sample number", 
        "sample proportion",   
    ]
    for lb in labels:
        titles += [f"posRate-{lb}", f"lift-{lb}"]
        titles += [f"posRate-{lb}-{col1}按照sample number分块来做", f"lift-{lb}"]

    for i, (aaa, ttl) in enumerate(zip(axes, titles)):
        aaa.set_title(ttl, fontproperties=myfont)
        # aaa.set_xticklabels(aaa.get_xticklabels(),rotation=30)
        aaa.set_yticklabels(aaa.get_yticklabels(),rotation=0) # 
        if (i == 0) or ("lift" in ttl):
            continue
        for t in aaa.texts: t.set_text(t.get_text() + " %")

    plt.suptitle(supTitle, fontproperties=myfont)# 
    
    # plt.show()
    
    plt.savefig(
        save_img_path, # os.path.join(save_img_path), 
        bbox_inches='tight'
    )
    
from IPython.display import display_html
from itertools import chain,cycle
    
def display_side_by_side(dfs, numSpaceIntervals = 3):
    html_str = ''
    for df in dfs:
        html_str += (df.render() + "&nbsp;" * numSpaceIntervals)
    display_html(
        html_str.replace('table','table style="display:inline"'), 
        raw=True
    )

def draw_level_score_heatmap_4(
    oot_label_score, labels, 
    col1 = "level", 
    col2 = "score", 
    first_subGraph_title = "rate_order",
    supTitle = "heatmap for data",
    n_bins = 5,
    save_img_path = "img.png",
    myfont = FontProperties(fname='SimHei.ttf'),
    figSize = (10, 10),
    color_range = [0,1],
):
    dfs = []
    
    oot_label_score["dummy"] = 1
    
    cross_7d_count = create_cross_table_count_2(
        oot_label_score[[col1, col2, "dummy"]], 
        col1, col2, "dummy",
        n_bins=n_bins
    )
    cross_7d_prop = create_cross_table_prop_2(
        oot_label_score[[col1, col2, "dummy"]], 
        col1, col2, "dummy",
        n_bins=n_bins
    )

    format_dict = {colName: '{:.2%}' for colName in cross_7d_count.columns}
    
    display_side_by_side([
        cross_7d_count.astype(int).style.background_gradient(axis=None, cmap="YlOrRd").set_caption("sample number"), 
        cross_7d_prop.style.format(format_dict).background_gradient(axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]).set_caption("sample proportion") # 
    ])
    
    dfs_posRate, dfs_lift = [], []
    
    for i, label in enumerate(labels):
        lb = label
        
#         cross_7d_mean = create_cross_table_sum_2( 
#             oot_label_score[[col1, col2, label]], 
#             col1, col2, label,
#             n_bins=n_bins
#         )
#         dfs_posRate += [
#             cross_7d_mean.style.background_gradient(axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]).set_caption(f"posNum-{lb}"),
#         ]        
        cross_7d_mean = create_cross_table_mean_2(
            oot_label_score[[col1, col2, label]], 
            col1, col2, label,
            n_bins=n_bins
        )
        dfs_posRate += [
            cross_7d_mean.style.format(format_dict).background_gradient(axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]).set_caption(f"posRate-{lb}-交叉排序"),
        ]
        dfs_lift += [
            (cross_7d_mean / cross_7d_mean.iloc[-1, -1]).style.background_gradient(axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]).set_caption(f"lift-{lb}-交叉排序")
        ]
    
        
        ################################ col1是分数。      
        new_df = [([None,] * cross_7d_count.shape[1]) for i in range(cross_7d_count.shape[0])]
        oot_label_score_1 = oot_label_score.copy(deep=True)
        oot_label_score_1 = oot_label_score_1.sort_values([col1]).reset_index(drop=True)
        ## 核心部分
        total_num = 0
        for h in range(cross_7d_count.shape[0] - 1):
            for j in range(cross_7d_count.shape[1] - 1):
                cur_num = cross_7d_count.iloc[h, j]
                new_df[h][j] = oot_label_score_1.loc[total_num: total_num + cur_num - 1, label].mean()
                total_num += cur_num
        ## 底下一行
        total_num = 0
        for j in range(cross_7d_count.shape[1] - 1):
            cur_num = cross_7d_count.iloc[-1, j]
            new_df[cross_7d_count.shape[0] - 1][j] = oot_label_score_1.loc[total_num: total_num + cur_num - 1, label].mean()
            total_num += cur_num
        ## 右边一列
        total_num = 0
        for h in range(cross_7d_count.shape[0] - 1):
            cur_num = cross_7d_count.iloc[h, -1]
            new_df[h][cross_7d_count.shape[1] - 1] = oot_label_score_1.loc[total_num: total_num + cur_num - 1, label].mean()
            total_num += cur_num
        ## 右下角
        total_num = 0
        cur_num = cross_7d_count.iloc[-1, -1]
        
#         new_df[-1][-1] = oot_label_score_1.loc[total_num: total_num + cur_num - 1, label].sum()
        new_df[-1][-1] = oot_label_score_1.loc[total_num: total_num + cur_num - 1, label].mean()

        total_num += cur_num
        cross_7d_mean_col1 = pd.DataFrame(new_df, columns = cross_7d_count.columns, index=cross_7d_count.index)

#         dfs_posRate += [
#             cross_7d_mean_col1.style.background_gradient(axis=None, cmap="YlOrRd").set_caption(f"posNum-{lb}-{col1}按照sample number分块来做"),
#         ]
        dfs_posRate += [
            cross_7d_mean_col1.style.format(format_dict).background_gradient(axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]).set_caption(f"posRate-{lb}-纯{col1}排序"),
        ]
        dfs_lift += [
            (cross_7d_mean_col1 / cross_7d_mean_col1.iloc[-1, -1]).style.background_gradient(axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]).set_caption(f"lift-{lb}-纯{col1}排序")
        ]
        
    display_side_by_side(dfs_posRate + dfs_lift)
    
    
    
def draw_level_score_heatmap_6(
    oot_label_score, labels, 
    col1 = "level", 
    col2 = "score", 
    n_bins = 5,
    myfont = FontProperties(fname='../SimHei.ttf'),
    figSize = (10, 10),
    color_range = [0,1],
):
    dfs = []
    
    oot_label_score["dummy"] = 1
    
    cross_7d_count = create_cross_table_count_1(
        oot_label_score[[col1, col2, "dummy"]], 
        col1, col2, "dummy",
        n_bins=n_bins
    )
    cross_7d_prop = create_cross_table_prop_1(
        oot_label_score[[col1, col2, "dummy"]], 
        col1, col2, "dummy",
        n_bins=n_bins
    )

    format_dict = {colName: '{:.2%}' for colName in cross_7d_count.columns}
    
    display_side_by_side([
        cross_7d_count.astype(int).style.background_gradient(axis=None, cmap="YlOrRd").set_caption("sample number"), 
        cross_7d_prop.style.format(format_dict).background_gradient(axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]).set_caption("sample proportion") # 
    ])
    
    
    def addNoCrossingRate(col1):
        ## 单纯地将col1分成n_bins**2个区间，看各个区间的平均下单率。
        oot_label_score.loc[:,f"{col1}_bin"] = pd.qcut(oot_label_score[col1], q=n_bins**2)
        tmp_table_col1 = pd.DataFrame(
            oot_label_score.groupby([f"{col1}_bin"])[label].mean()
        )
        idx = tmp_table_col1.index
        tmp_table_col1 = tmp_table_col1.append(
            {
                label: oot_label_score[label].mean()
            },
            ignore_index=True
        )
        tmp_table_col1.index=list(idx) + ["total"]
        tmp_table_col1_lift = tmp_table_col1 / tmp_table_col1.iloc[-1,-1]
        return (
            ############
            tmp_table_col1.reset_index().style.format(
                format_dict
            ).background_gradient(
                axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]
            ).set_caption(f"posRate-{label}-{col1}单列分bin"), 
            ##############
            tmp_table_col1_lift.reset_index().style.format(
                format_dict
            ).background_gradient(
                axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]
            ).set_caption(f"lift-{label}-{col1}单列分bin")
       )
    
    for i, label in enumerate(labels):
        dfs_posRate, dfs_lift = [], []
        
        lb = label
        
        cross_7d_mean = create_cross_table_mean_1(
            oot_label_score[[col1, col2, label]], 
            col1, col2, label,
            n_bins=n_bins
        )
        dfs_posRate += [
            cross_7d_mean.style.format(format_dict).background_gradient(axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]).set_caption(f"posRate-{lb}-交叉排序"),
        ]
        dfs_lift += [
            (cross_7d_mean / cross_7d_mean.iloc[-1, -1]).style.background_gradient(axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]).set_caption(f"lift-{lb}-交叉排序")
        ]        
            
        x1, x2 = addNoCrossingRate(col1)
        dfs_posRate += [x1, ]
        dfs_lift += [x2, ]
        
        x1, x2 = addNoCrossingRate(col2)
        dfs_posRate += [x1, ]
        dfs_lift += [x2, ]
        
        display_side_by_side(dfs_posRate + dfs_lift)
    
    
    
def easy_bin_posRate(
    cross, score_names, labels, 
    convert_nBins = 5,
    duplicates="drop", 
):
    '''
    cross: 综合表。包括了score_names所代表的各种分的名字，以及labels代表的各种label的名字。
    convert_nBins：分箱，分几箱。
    '''
    format_dict = {colName: '{:.2%}' for colName in score_names}
    for label in labels:
        tbs_1 = []         
        rst = {}
        for colName in score_names:
            bin_col_name = f"{colName}-bin"
            cross.loc[:, bin_col_name], short_score_bin = pd.qcut(
                cross[colName], q=convert_nBins, retbins=True, labels=False, 
                duplicates = duplicates
            )
            group_new = cross.groupby(bin_col_name)
            v1_bin = group_new[label].sum()/group_new[label].count()
            rst[colName] = v1_bin   
        tbs_1.append(
            (
                pd.DataFrame(rst)
                .style
                .format(format_dict)
                .bar(
                    align='zero'
                )
                .set_caption(
                    "label: {}; #Sample: {}; posRate: {}".format(
                        label, 
                        cross.shape[0], 
                        cross[label].mean()
                    )
                )
            )
        )
        display_side_by_side(tbs_1)
        
def easy_bin_posRate_1(
    cross, score_names, labels, 
    convert_nBins = 5,
    display_binNum = False,
    duplicates="drop", 
):
    '''
    相比于easy_bin_posRate的改进：
        1. 增加了duplicates，可以设置qcut的同名参数。这个有什么用？比如遇到分bin界限有重复的，要如何处理。
        2. 增加了分箱数量的展示。分箱数量会单独展示为一个表。display_binNum为True就是要显示，否则就是不展示。默认不展示。
    cross: 综合表。包括了score_names所代表的各种分的名字，以及labels代表的各种label的名字。
    convert_nBins：分箱，分几箱。
    '''
    format_dict = {colName: '{:.2%}' for colName in score_names}
    for label in labels:
        tbs_1 = [] 
        rst = {} ## 记录分bin下单结果。
        rst_cnt = {} ## 记录分bin数量。
        # rst_binEdge = {}
        for colName in score_names:
            bin_col_name = f"{colName}-bin"
            cross.loc[:, bin_col_name], short_score_bin = pd.qcut(
                cross[colName], q=convert_nBins, retbins=True, labels=False, 
                duplicates = duplicates
            )
            group_new = cross.groupby(bin_col_name)
            rst[colName] = group_new[label].sum()/group_new[label].count()
            rst_cnt[colName] = group_new[label].count()
            # print(group_new[label].count())
            # rst_binEdge[colName] = short_score_bin
        tbs_1.append(
            (
                pd.DataFrame(rst)
                .style
                .format(format_dict)
                .bar(
                    align='zero'
                )
                .set_caption(
                    "bin rate ||| label: {}; #Sample: {}; posRate: {}".format(
                        label, 
                        cross.shape[0], 
                        cross[label].mean()
                    )
                )
            )
        )
        if display_binNum:
            tbs_1.append(
                (
                    pd.DataFrame(rst_cnt)
                    .style
                    .bar(
                        align='zero'
                    )
                    .set_caption(
                        "bin count ||| label: {}; #Sample: {}; posRate: {}".format(
                            label, 
                            cross.shape[0], 
                            cross[label].mean()
                        )
                    )
                )
            )
        
        display_side_by_side(tbs_1)
        
def bin_by_scr_and_see_feaVal_distribution(
    cross,
    scr_col,
    feas_figure,
    convert_nBins = 10,
    color_range = [0,1]
):
    '''
    cross是一个表，把评分和特征都整合在一起的表。
    根据scr_col这个分数进行分bin。
    然后在各个分bin里统计feas_figure这些特征的一些统计值："平均数", "最小值", "最大值", "众数"。
    这个的结果可以视作用户画像。
    '''
    tbs = []
    format_dict = {colName: '{:.2f}' for colName in feas_figure}
    bin_col_name = f"{scr_col}-bin"
    cross.loc[:, bin_col_name] = pd.qcut(
        cross[scr_col], q=convert_nBins, # retbins=True, labels=False
    )
    group_new = cross.groupby(bin_col_name)

    for calc_type, calc_name in zip(
        ["mean()", "min()", "max()", "apply(lambda x:x.mode())"],
        ["平均数", "最小值", "最大值", "众数"]
    ): ## 我们做这个是为了做一些不一样的统计。
        rst = {}
        for fea in feas_figure:
            rst[fea] = eval(f"group_new[fea].{calc_type}")
        tbs.append(
            pd.DataFrame(rst).reset_index().style.format(format_dict).background_gradient(
                axis=0, cmap="YlOrRd", low=color_range[0], high=color_range[1]
            ).set_caption(f"分bin后各bin的：{calc_name}")
        )
    display_side_by_side(tbs)
    
    
def create_cross_table_count_2(df, levelName, colName2, labelColName, n_bins=5, duplicates = "drop"):
    '''
    colName2 will be horizontal axis, colName1 will be vertical axis.
    计算一个格子里样本总数。
    如果分箱边界重复了，也就是没法等频分，采用drop的方法将多余的分箱边界删除。
    '''
    colName1 = f'{levelName}-bin'
    df.loc[:,colName1] = pd.qcut(df[f'{levelName}'], q=n_bins, duplicates = duplicates) # pd.cut(df[f'{levelName}'], bins=n_bins) # 
    df.loc[:,f'{colName2}-bin'] = pd.qcut(df[f'{colName2}'], q=n_bins, duplicates = duplicates) # pd.cut(df[f'{colName2}'], bins=n_bins) # 
    
    # display(df)
    
    tmp = df.groupby([f'{colName1}', f'{colName2}-bin'])[f"{labelColName}"].count().unstack() 
    
    #######################################################################
    overall1 = df.groupby([f'{colName1}'])[f"{labelColName}"].count()
    tmp = pd.concat(
        [
            pd.DataFrame(
#                 {f"{levelName}-total": overall1}
                {f"total": overall1}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    #######################################################################
    overall2 = df.groupby([f'{colName2}-bin'])[f"{labelColName}"].count()
    overall2.index
    
    tmp = tmp.T
    
    tmp = pd.concat(
        [
            pd.DataFrame(
#                 {f"{colName2}-total": overall2}
                {f"total": overall2}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    tmp = tmp.T
    
    tmp.iloc[-1, -1] = df.shape[0]
    
    return tmp 

def create_cross_table_mean_2(df, levelName, colName2, labelColName, n_bins=5, duplicates = "drop"):
    '''
    colName2 will be horizontal axis, colName1 will be vertical axis.
    计算一个格子里样本的下单率。
    如果分箱边界重复了，也就是没法等频分，采用drop的方法将多余的分箱边界删除。
    '''
    colName1 = f'{levelName}-bin'
    df.loc[:,colName1] = pd.qcut(df[f'{levelName}'], q=n_bins, duplicates = duplicates) # pd.cut(df[f'{levelName}'], bins=n_bins) # 
    df.loc[:,f'{colName2}-bin'] = pd.qcut(df[f'{colName2}'], q=n_bins, duplicates = duplicates) # pd.cut(df[f'{colName2}'], bins=n_bins) # 
    
    tmp = df.groupby([f'{colName1}', f'{colName2}-bin'])[f"{labelColName}"].mean().unstack() 
    
    #######################################################################
    overall1 = df.groupby([f'{colName1}'])[f"{labelColName}"].mean()
    tmp = pd.concat(
        [
            pd.DataFrame(
#                 {f"{levelName}-total": overall1}
                {f"total": overall1}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    #######################################################################
    overall2 = df.groupby([f'{colName2}-bin'])[f"{labelColName}"].mean()
    overall2.index
    
    tmp = tmp.T
    
    tmp = pd.concat(
        [
            pd.DataFrame(
#                 {f"{colName2}-total": overall2}
                {f"total": overall2}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    tmp = tmp.T
    
    tmp.iloc[-1, -1] = df[labelColName].mean() # df.shape[0]
    
    return tmp 

def create_cross_table_prop_2(df, levelName, colName2, labelColName, n_bins=5, duplicates = "drop"):
    '''
    colName2 will be horizontal axis, colName1 will be vertical axis.
    计算一个格子里的样本数占总样本数的比例。
    如果分箱边界重复了，也就是没法等频分，采用drop的方法将多余的分箱边界删除。
    '''
    colName1 = f'{levelName}-bin'
    df.loc[:,colName1] = pd.qcut(df[f'{levelName}'], q=n_bins, duplicates = duplicates) # pd.cut(df[f'{levelName}'], bins=n_bins) # 
    df.loc[:,f'{colName2}-bin'] = pd.qcut(df[f'{colName2}'], q=n_bins, duplicates = duplicates) # pd.cut(df[f'{colName2}'], bins=n_bins) # 
    
    tmp = df.groupby([f'{colName1}', f'{colName2}-bin'])[f"{labelColName}"].count().unstack() / df.shape[0]
    
    #######################################################################
    overall1 = df.groupby([f'{colName1}'])[f"{labelColName}"].count() / df.shape[0]
    tmp = pd.concat(
        [
            pd.DataFrame(
#                 {f"{levelName}-total": overall1}
                {f"total": overall1}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    #######################################################################
    overall2 = df.groupby([f'{colName2}-bin'])[f"{labelColName}"].count() / df.shape[0]
    overall2.index
    
    tmp = tmp.T
    
    tmp = pd.concat(
        [
            pd.DataFrame(
#                 {f"{colName2}-total": overall2}
                {f"total": overall2}
            ),
            pd.DataFrame(tmp), 
        ], axis=1
    )
    tmp = tmp[
        list(tmp.columns[1:]) + list(tmp.columns[:1])
    ]
    tmp = tmp.T
    
    tmp.iloc[-1, -1] = 1
    
    return tmp 

def draw_level_score_heatmap_7(
    oot_label_score, labels, 
    col1 = "level", 
    col2 = "score", 
    n_bins = 5,
    myfont = FontProperties(fname='../SimHei.ttf'),
    figSize = (10, 10),
    color_range = [0,1],
):
    '''
    输入的数据oot_label_score包含这些列：结清距今天数，用来划分人群；各个评分；各个label；trace_id。
    跟draw_level_score_heatmap_6的区别在于，单分分箱是以交叉分箱的边界来划分的。这样单分分箱的结果和交叉分箱的结果就有可比性了；
    draw_level_score_heatmap_6 单分分箱是按照分箱边界来搞的
    '''
    dfs = []
    
    oot_label_score["dummy"] = 1
    
    cross_7d_count = create_cross_table_count_2(
        oot_label_score[[col1, col2, "dummy"]], 
        col1, col2, "dummy",
        n_bins=n_bins
    )
    cross_7d_prop = create_cross_table_prop_2(
        oot_label_score[[col1, col2, "dummy"]], 
        col1, col2, "dummy",
        n_bins=n_bins
    )

    format_dict = {colName: '{:.2%}' for colName in cross_7d_count.columns}
    
    display_side_by_side([
        cross_7d_count.astype(int).style.background_gradient(axis=None, cmap="YlOrRd").set_caption("sample number"), 
        cross_7d_prop.style.format(format_dict).background_gradient(axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]).set_caption("sample proportion") # 
    ])
    
#     def addNoCrossingRate(col1, bin_num):
#         '''单纯地将col1分成n_bins**2个区间，看各个区间的平均下单率。这种切分方法不严谨，不建议使用。'''
#         oot_label_score.loc[:,f"{col1}_bin"] = pd.qcut(oot_label_score[col1], q=bin_num, duplicates = "drop")
#         tmp_table_col1 = pd.DataFrame(
#             oot_label_score.groupby([f"{col1}_bin"])[label].mean()
#         )
#         idx = tmp_table_col1.index
#         tmp_table_col1 = tmp_table_col1.append(
#             {
#                 label: oot_label_score[label].mean()
#             },
#             ignore_index=True
#         )
#         tmp_table_col1.index=list(idx) + ["total"]
#         tmp_table_col1_lift = tmp_table_col1 / tmp_table_col1.iloc[-1,-1]
#         return (
#             ############
#             tmp_table_col1.reset_index().style.format(
#                 format_dict
#             ).background_gradient(
#                 axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]
#             ).set_caption(f"posRate-{label}-{col1}单列分bin"), 
#             ##############
#             tmp_table_col1_lift.reset_index().style.format(
#                 format_dict
#             ).background_gradient(
#                 axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]
#             ).set_caption(f"lift-{label}-{col1}单列分bin")
#        )
    
    def addNoCrossingRate(col1, label):
        '''
        按照cross_7d_count的各个小箱的边界去切分单分。
        '''
        ################################ col1是分数。      
        new_df = [([None,] * cross_7d_count.shape[1]) for i in range(cross_7d_count.shape[0])]
        oot_label_score_1 = oot_label_score.copy(deep=True)
        oot_label_score_1 = oot_label_score_1.sort_values([col1]).reset_index(drop=True)
        ## 核心部分
        total_num = 0
        for h in range(cross_7d_count.shape[0] - 1):
            for j in range(cross_7d_count.shape[1] - 1):
                cur_num = cross_7d_count.iloc[h, j]
                new_df[h][j] = oot_label_score_1.loc[total_num: total_num + cur_num - 1, label].mean()
                total_num += cur_num
        ## 底下一行
        total_num = 0
        for j in range(cross_7d_count.shape[1] - 1):
            cur_num = cross_7d_count.iloc[-1, j]
            new_df[cross_7d_count.shape[0] - 1][j] = oot_label_score_1.loc[total_num: total_num + cur_num - 1, label].mean()
            total_num += cur_num
        ## 右边一列
        total_num = 0
        for h in range(cross_7d_count.shape[0] - 1):
            cur_num = cross_7d_count.iloc[h, -1]
            new_df[h][cross_7d_count.shape[1] - 1] = oot_label_score_1.loc[total_num: total_num + cur_num - 1, label].mean()
            total_num += cur_num
        ## 右下角
        total_num = 0
        cur_num = cross_7d_count.iloc[-1, -1]
#         new_df[-1][-1] = oot_label_score_1.loc[total_num: total_num + cur_num - 1, label].sum()
        new_df[-1][-1] = oot_label_score_1.loc[total_num: total_num + cur_num - 1, label].mean()

        total_num += cur_num
        cross_7d_mean_col1 = pd.DataFrame(new_df, columns = cross_7d_count.columns, index=cross_7d_count.index)
        cross_7d_mean_col1_lift = cross_7d_mean_col1 / cross_7d_mean_col1.iloc[-1,-1]
        
        return (
            ############
            cross_7d_mean_col1.reset_index().style.format(
                format_dict
            ).background_gradient(
                axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]
            ).set_caption(f"posRate-{label}-{col1}单列分bin"), 
            ##############
#             .format(
#                 format_dict
#             )
            cross_7d_mean_col1_lift.reset_index().style.background_gradient(
                axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]
            ).set_caption(f"lift-{label}-{col1}单列分bin")
       )
    
    for i, label in enumerate(labels):
        dfs_posRate, dfs_lift = [], []
        
        lb = label
        
        cross_7d_mean = create_cross_table_mean_2(
            oot_label_score[[col1, col2, label]], 
            col1, col2, label,
            n_bins=n_bins
        )
#         dfs_posRate += [
#             cross_7d_mean.style.format(format_dict).background_gradient(axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]).set_caption(f"posRate-{lb}-交叉排序"),
#         ]
#         dfs_lift += [
#             (cross_7d_mean / cross_7d_mean.iloc[-1, -1]).style.background_gradient(axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]).set_caption(f"lift-{lb}-交叉排序")
#         ]        
            
#         x1, x2 = addNoCrossingRate(col1, label) # (cross_7d_mean.shape[0] - 1) * (cross_7d_mean.shape[1] - 1) 
#         dfs_posRate += [x1, ]
#         dfs_lift += [x2, ]
        
#         x1, x2 = addNoCrossingRate(col2, label) # (cross_7d_mean.shape[0] - 1) * (cross_7d_mean.shape[1] - 1) 
#         dfs_posRate += [x1, ]
#         dfs_lift += [x2, ]
        
#         display_side_by_side(dfs_posRate + dfs_lift)
        
        '''
        相较于 draw_level_score_heatmap_6 , 这里将数据表重新排列了，也就是把每个分数各自的posRate和lift放到一起。
        上面的一大块注释，能把各个分数的posRate放到一起，lift也放到一起。
        '''
        dfs = [
            cross_7d_mean.style.format(format_dict).background_gradient(axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]).set_caption(f"posRate-{lb}-交叉排序"),
            (cross_7d_mean / cross_7d_mean.iloc[-1, -1]).style.background_gradient(axis=None, cmap="YlOrRd", low=color_range[0], high=color_range[1]).set_caption(f"lift-{lb}-交叉排序"),
        ] + list(
            addNoCrossingRate(col1, label)
        ) + list(
            addNoCrossingRate(col2, label)
        )
        display_side_by_side(dfs)
        
def output_as_fig(text, fontSize = 30, fontPath = '../../SimHei.ttf'):
    '''
    将text变成一张图片，然后输出。
    这么做没有什么特别用意，目的就在于能够让输出变得好看一些。
    '''
    # 创建一张空白的图片
    img = Image.new('RGB', (len(text)*fontSize, fontSize), color = (255, 255, 255))
    
    # 在图片中添加文本
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(fontPath, fontSize)
    textwidth, textheight = draw.textsize(text, font)

    x = (img.width - textwidth) / 2
    y = (img.height - textheight) / 2

    draw.text((x, y), text, font=font, fill=(0, 0, 0))

    # 在 Jupyter Notebook 中显示图片
    display(img)
    
def easy_bin_posRate_2(
    cross, score_names, labels, 
    convert_nBins = 5,
    display_binNum = False,
    display_lift_chart = False,
    duplicates="drop", 
):
    '''
    相比于easy_bin_posRate_1的改进：
        可以算lift值。
    cross: 综合表。包括了score_names所代表的各种分的名字，以及labels代表的各种label的名字。
    convert_nBins：分箱，分几箱。
    '''
    format_dict = {colName: '{:.2%}' for colName in score_names}
    for label in labels:
        tbs_1 = [] 
        rst = {} ## 记录分bin下单结果。
        rst_cnt = {} ## 记录分bin数量。
        # rst_binEdge = {}
        for colName in score_names:
            bin_col_name = f"{colName}-bin"
            cross.loc[:, bin_col_name], short_score_bin = pd.qcut(
                cross[colName], q=convert_nBins, retbins=True, labels=False, 
                duplicates = duplicates
            )
            group_new = cross.groupby(bin_col_name)
            rst[colName] = group_new[label].sum()/group_new[label].count()
            rst_cnt[colName] = group_new[label].count()
            # print(group_new[label].count())
            # rst_binEdge[colName] = short_score_bin
        posRate_table = pd.DataFrame(rst)
        tbs_1.append(
            (
                posRate_table
                .style
                .format(format_dict)
                .bar(
                    align='zero'
                )
                .set_caption(
                    "bin rate ||| label: {}; #Sample: {}; posRate: {}".format(
                        label, 
                        cross.shape[0], 
                        cross[label].mean()
                    )
                )
            )
        )        
        
        if display_lift_chart: 
            lift_table = posRate_table / cross[label].mean()
            tbs_1.append(
                (
                    lift_table
                    .style
                    .bar(
                        align='zero',
                        color="cyan"
                    )
                    .set_caption(
                        "lift ||| label: {}; #Sample: {}; posRate: {}".format(
                            label, 
                            cross.shape[0], 
                            cross[label].mean()
                        )
                    )
                )
            )
        
        if display_binNum:
            cnt_table = pd.DataFrame(rst_cnt)
            tbs_1.append(
                (
                    cnt_table
                    .style
                    .bar(
                        align='zero'
                    )
                    .set_caption(
                        "bin count ||| label: {}; #Sample: {}; posRate: {}".format(
                            label, 
                            cross.shape[0], 
                            cross[label].mean()
                        )
                    )
                )
            )
        
        display_side_by_side(tbs_1)