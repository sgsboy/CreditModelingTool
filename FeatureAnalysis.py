# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:26:36 2017

@author: sgs

About: feature analysis
"""
# 加载包
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os


# 1、特征分析类函数
class FeatureAnalysis:
    # 初始化参数
    def __init__(self,df,y,value_num=20,precision=0,char_num=20,plot_num=20,dum_beta=0.01,\
                     not_nan_beta=0.01,max_value_beta=0.99):
        self.df = df
        self.y = y
        self.value_num = value_num
        self.precision = precision
        self.char_num = char_num
        self.dum_beta = dum_beta
        self.plot_num=plot_num
        self.not_nan_beta=not_nan_beta
        self.max_value_beta=max_value_beta
           
    # 最大值计数  
    def max_value_cnt(self,x):
        if len(x.dropna())>0:
            return max(x.groupby(x).count())
        else:
            return 0  
          
    # 单变量T检验
    def sigel_ttest_pvalue(self,x,y):
        if x.dtypes!='object':
            res=scipy.stats.ttest_ind(x,y).pvalue
            return res.astype(np.float64)
        elif x.dtypes=='object':
            return 1.00
             
    # 变量描述（总体）      
    ## 统计df的缺失度、集中度、类型、最大最小均值中位数、与因变量的p值等      
    def desc_df(self):     
        ''' 
        #测试代码
        df=model_data;y='Y_LABEL'
        '''
        # 入参
        df=self.df
        y=self.y
        # 变量类型
        df_dtype=DataFrame(df.dtypes,columns=['dtypes'])
        df_dtype['count']= len(df)
        df_dtype['var']=df_dtype.index
        int_var=list(df_dtype[df_dtype['dtypes']!='object'].loc[:,'var'])
        #  取值、非空值、集中值
        df_unique_num=DataFrame(df.apply(lambda x: len(x.unique())),columns=['unique_num'])
        df_not_nan_num=DataFrame(df.apply(lambda x: len(x.dropna())),columns=['not_nan_num'])
        df_not_nan_num['not_nan_num_beta']=df_not_nan_num['not_nan_num']/ len(df)
        df_max_value_cnt=DataFrame(df.apply(self.max_value_cnt),columns=['max_value_cnt_num'])
        df_max_value_cnt['max_value_cnt_num_beta']=df_max_value_cnt['max_value_cnt_num']/ len(df)
        # p值
        p_value_index=[];p_value_value=[]
        for i in df.columns:
            p_value_index.append(i);
            p_value_value.append(self.sigel_ttest_pvalue(df[i],df[y]))
            # woe值
        df_woe=self.iv_value_df()
        # 数据合并
        df_p_value=DataFrame(p_value_value,index=p_value_index,columns=['p_value'])
        df_desc=pd.merge(df_dtype,DataFrame(df.loc[:,int_var].describe().T).drop('count',1),left_index=True,right_index=True,how='left')
        df_desc=pd.merge(df_desc,df_unique_num,left_index=True,right_index=True,how='left')
        df_desc=pd.merge(df_desc,df_not_nan_num,left_index=True,right_index=True,how='left')
        df_desc=pd.merge(df_desc,df_max_value_cnt,left_index=True,right_index=True,how='left')
        df_desc=pd.merge(df_desc,df_p_value,left_index=True,right_index=True,how='left')
        df_desc=pd.merge(df_desc,df_woe,left_index=True,right_index=True,how='left')
        # 结果返回
        return df_desc
    
        
    # 字符型变量取值描述
    def char_var_desc(self):
        ''' 
        #测试代码
        df_desc=desc_df(df,y);char_num=20
        '''
        # 入参
        df=self.df
        char_num=self.char_num
        #
        df_desc=self.desc_df()
        # 非数值变量提取
        df_desc_object=df_desc[df_desc['dtypes']=='object'] 
        # 小于num个值的非数值型变量进行dummies，即生产哑变量
        df_desc_object_dum=df_desc_object[df_desc_object['unique_num']<char_num]
        # 获取字符型变量
        df_desc_object_dum_index=[];df_desc_object_dum_value=[]
        for i in list(df_desc_object_dum.index):
            df_desc_object_dum_index.append(i)
            df_desc_object_dum_value.append("".join(str(list(df[i].unique()))))
        df_desc_object_dum_value=DataFrame(df_desc_object_dum_value,index=df_desc_object_dum_index,columns=['values'])
        # 合并    
        df_desc_object_res=df_desc_object[["dtypes","unique_num"]].join(df_desc_object_dum_value,how='left')
        df_desc_object_res['values']=df_desc_object_res['values'].fillna("取值过多剔除")
        df_desc_object_res=df_desc_object_res.sort_values(by=['unique_num'],ascending=True)
        return df_desc_object_res
    
    # 字符型变量剔除
    def char_var_dum(self):
        ''' 
        #测试代码
        df=model_data;df_desc=model_data_desc;dum_beta=0.01;char_num=20
        '''  
        # 入参
        df=self.df
        char_num=self.char_num
        dum_beta=self.dum_beta
        #
        df_desc=self.desc_df()
        df_desc_object=df_desc[df_desc['dtypes']=='object'] 
        df_desc_object_del=df_desc_object[df_desc_object['unique_num']>=char_num]
        df=df.drop(list(df_desc_object_del.index),1)
        df_desc_object_dum=df_desc_object[df_desc_object['unique_num']<char_num]
        # 获取字符型变量
        for i in list(df_desc_object_dum.index):
            df_dum=pd.get_dummies(df[i].fillna("nan"),prefix=i)
            for j in df_dum.columns:
                if len(df_dum[df_dum[j]==1])/len(df_dum)<=dum_beta or len(df_dum[df_dum[j]==1])/len(df_dum)>=(1-dum_beta):
                    df_dum=df_dum.drop(j,1)
            df=df.join(df_dum,how='left')
            df=df.drop(i,1)
        # 返回修改后的
        return df  
         
    # iv计算 
    def iv_value(self,var_df):
        #入参
        precision=self.precision
        #
        var_df=var_df.fillna(0)
        var_0_sum=var_df.iloc[:,0].sum()
        var_1_sum=var_df.iloc[:,1].sum()
        var_0_sum_beta=[]
        for i in range(len(var_df)):
            var_0_sum_beta.append(var_df.iloc[i,:].sum()/(var_0_sum+var_1_sum))
        var_df['num_bate']=var_0_sum_beta
        var_df['bad_bate']=var_df.iloc[:,1]/var_1_sum
        var_df['good_bate']=var_df.iloc[:,0]/var_0_sum
        var_df['bad_good_bate']=var_df['bad_bate']-var_df['good_bate']
        var_df['overdue_bate']=var_df.iloc[:,1]/(var_df.iloc[:,0]+var_df.iloc[:,1])
        var_df['woe']=np.log(((var_df.iloc[:,1]+precision)/var_1_sum)/((var_df.iloc[:,0]+precision)/var_0_sum))
        iv_value=sum(var_df['bad_good_bate']*var_df['woe'])
        return(iv_value,var_df)
           
    # iv值计算----df
    def iv_value_df(self,file_name='model_data_1',file_on=False,plot_on=False):
        '''
        #测试代码
        df=model_data;y='﻿y_value';value_num=20;precision=0;plot_num=20
        '''
        # 入参
        df=self.df
        y=self.y
        value_num=self.value_num
        plot_num=self.plot_num
        cur_dir=os.getcwd()
        #
        iv_indexs=[];
        iv_values=[];
        for var in df.columns:
            #
            var_df=DataFrame(df[var].groupby([df[var],df[y]]).count().unstack())
            nan_len= len(df[df[var].isnull()==True])
            #
            if nan_len<len(df) and nan_len>0:
                nan_true= df[df[var].isnull()==True].ix[:,[y]]==1
                nan_1=len(nan_true[nan_true[y]==True])
                var_nan=DataFrame([nan_len-nan_1],index=[-1])
                var_nan[1]=[nan_1]
                var_nan.columns=[0, 1]
                var_df=pd.concat([var_df,var_nan], axis=0)
            #    
            if len(var_df.columns)>=2 and len(var_df)>=2 and df[var].dtypes!='object' and var!=y:
                print(var) 
                if len(var_df)>=value_num:
                    var_y_df=df.loc[:,[y,var]]
                    var_y_df=var_y_df.fillna(-1) 
                    var_y_df=var_y_df.sort_values(by=[var],ascending=True)
                    var_y_df['index']=list(range(1,len(var_y_df)+1))
                    var_y_df['index_cut']=pd.cut(var_y_df['index'],value_num,labels=list(range(1,value_num+1)))
                    # 初始分组
                    for index_cut in range(2,len(var_y_df['index_cut'].unique())+1):
                        cut_min_now=var_y_df[var_y_df['index_cut']==index_cut][var].min()
                        cut_min_now_num=len(var_y_df.loc[(var_y_df['index_cut']==index_cut) &(var_y_df[var]==cut_min_now)])
                        cut_max_last=var_y_df[var_y_df['index_cut']<index_cut][var].max()
                        cut_max_last_num=len(var_y_df.loc[(var_y_df['index_cut']<index_cut) &(var_y_df[var]==cut_max_last)])
                        if cut_min_now==cut_max_last and cut_min_now_num>=cut_max_last_num:
                            var_y_df['index_cut'].loc[(var_y_df['index_cut']==index_cut-1) &(var_y_df[var]==cut_min_now)]=index_cut
                        elif cut_min_now==cut_max_last and cut_min_now_num<cut_max_last_num :
                            index_cut_now=var_y_df['index_cut'].loc[(var_y_df[var]==cut_min_now)].min()
                            var_y_df['index_cut'].loc[(var_y_df['index_cut']==index_cut) &(var_y_df[var]==cut_min_now)]=index_cut_now
                    list_unique= list(var_y_df['index_cut'].unique())
                    for i in range(0,len(list_unique)):
                        var_y_df['index_cut'].loc[(var_y_df['index_cut']==list_unique[i])]=i+1
                    # 分组结果保存
                    var_df=DataFrame(var_y_df['index_cut'].groupby([var_y_df['index_cut'],var_y_df[y]]).count().unstack())                        
                # woe 计算
                value,var_df=self.iv_value(var_df)
                # iv save
                iv_indexs.append(var)
                iv_values.append(value)
                # index save
                if file_on==True:
                    file_1=os.path.join(cur_dir,file_name)
                    if os.path.exists(file_1)==False:
                        os.mkdir(file_1)
                    file_2=os.path.join(file_1,"woe_value")
                    if os.path.exists(file_2)==False:
                        os.mkdir(file_2)
                    var_df.to_csv(file_2+'\\'+var+'.csv')
             
                # plot save
                if plot_on==True:
                    plot_1=os.path.join(cur_dir,file_name)
                    if os.path.exists(plot_1)==False:
                        os.mkdir(plot_1)
                    plot_2=os.path.join(plot_1,"woe_risk_plot")
                    if os.path.exists(plot_2)==False:
                        os.mkdir(plot_2)
                    if len(var_df)<=plot_num:
                        fig=plt.figure()
                        ax=fig.add_subplot(1,1,1)
                        ax.scatter(var_df.index,var_df['overdue_bate'],s=var_df['num_bate']*10000)                 
                        for xx,yy,zz in zip(var_df.index,var_df['overdue_bate'],var_df['num_bate']):
                            plt.annotate('%s' %round(zz,2),xy=(xx,yy+0.02),xytext=(xx,yy),textcoords='offset points',ha='center',va='top')
                            plt.annotate('%s' %round(yy,2),xy=(xx,yy-0.02),xytext=(xx,yy),textcoords='offset points',ha='center',va='top') 
                            plt.xlabel(var)  
                            plt.ylabel('Y')  
                            plt.grid(True) 
                            plt.savefig(plot_2+'\\'+var+'.pdf')
                            plt.draw()
                            #将数据详情表添加
                            if  len(var_df)<=3:
                                the_table = plt.table(cellText=var_df.round(4).values,\
                                                colWidths = [0.1]*len(var_df.columns),\
                                                rowLabels=var_df.index,\
                                                colLabels=var_df.columns,\
                                                loc=1)
                                the_table.auto_set_font_size(False)
                                the_table.set_fontsize(4)
                    
                    else:
                        var_y_df=df.loc[:,[y,var]]
                        var_y_df=var_y_df.fillna(-1)
                        var_y_df=var_y_df.sort_values(by=[var],ascending=True)
                        var_y_df=var_y_df.reset_index(drop=True)
                        var_y_df['index']=var_y_df.index
                        var_y_df['lable']=1
                        cut_len=int(len(var_y_df)/plot_num)+1
                        for i in range(plot_num):
                            var_y_df['lable'].loc[(var_y_df['index']>=(cut_len*(i)))&(var_y_df['index']<(cut_len*(i+1)))]=i+1
                        var_y_df_res=var_y_df.groupby('lable').agg(['sum','count'])
                        var_y_df_res.columns=['y_sum','count',var,'count_1','index','count_2']
                        var_y_df_res['x']=var_y_df_res[var]/var_y_df_res['count']
                        var_y_df_res['y_beta']=var_y_df_res['y_sum']/var_y_df_res['count']
                
                        fig=plt.figure()
                        ax=fig.add_subplot(1,1,1)
                        plt.plot(var_y_df_res['x'], var_y_df_res['y_beta'], marker='o', mec='r', mfc='w')
                        plt.xlabel(var)  
                        plt.ylabel('Y')  
                        plt.grid(True) 
                        plt.savefig(plot_2+'\\'+var+'.pdf') 
                        plt.draw()      
            else:
                # iv save
                iv_indexs.append(var)
                iv_values.append(np.nan)
        # woe 结果保存
        df_iv_value=DataFrame(iv_values,index=iv_indexs,columns=['iv_value'])  
        return df_iv_value
        
    # 特征初选----df
    def var_pro_choose(self,df_desc):
        '''
        #测试代码
        df=model_data_1;y='y_value';df_desc=model_data_1_desc;not_nan_beta=0.01;max_value_beta=0.99
        '''
        # 入参
        df=self.df
        y=self.y
        not_nan_beta=self.not_nan_beta
        max_value_beta=self.max_value_beta
        #
        df_desc['var_name']=df_desc.index
        var_pro_choose=list(df_desc['var_name'].loc[(df_desc['not_nan_num_beta']>=not_nan_beta)\
                                                  & (df_desc['max_value_cnt_num_beta']<=max_value_beta)\
                                                  & (df_desc['unique_num']>=2)])
        if y not in var_pro_choose:
            var_pro_choose.append(y)
        df_pro=df.loc[:,var_pro_choose]
        return df_pro


    
# 2、特征分析类函数 应用

# 单变量泡泡图
def iv_value_var(df,var_list,y,file_name='md1',file_on=True,plot_on=True):
    fa_1=FeatureAnalysis(df=df.loc[:,var_list],y=y)
    df_iv_value=fa_1.iv_value_df(file_name=file_name,file_on=file_on,plot_on=plot_on) 
    return df_iv_value

