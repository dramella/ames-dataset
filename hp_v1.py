#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 18:04:47 2019

@author: Debora
"""

##Libraries
import pandas as pd
import numpy as np
import os
import seaborn as sns
from seaborn import countplot
import matplotlib as mp
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler , LabelEncoder
#EDA
inputpath='/Users/Debora/Desktop/DATA SCIENCE/HousePricesProject'
train_df=pd.read_csv(os.path.join(inputpath,'train.csv'))
test_df=pd.read_csv(os.path.join(inputpath,'test.csv'))


Y_train_df=train_df['SalePrice']
X_train_df=train_df.drop(labels=['SalePrice'], axis=1)

def correlation_heatmap(X_train_df):
    correlations = X_train_df.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show();
    
correlation_heatmap(X_train_df)

#Check correlations
# Create correlation matrix
corr_matrix = X_train_df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
for i in to_drop:
    del X_train_df[i]



####Handling Missing Values

#Dropping rows with missing value rate higher than threshold
threshold_rows=0.8
X_train_df = X_train_df.loc[X_train_df.isnull().mean(axis=1) < threshold_rows]

nas_list=X_train_df.isna().sum()/len(X_train_df)
nas_list=nas_list[nas_list>0]


high_threshold=0.8
medium_threshold=0.3
low_threshold=0.05

columns_with_medium_NAs=nas_list[nas_list<=medium_threshold]
columns_with_medium_NAs=columns_with_medium_NAs[columns_with_medium_NAs>low_threshold]


columns_with_high_NAs=nas_list[nas_list>=high_threshold]
for i in columns_with_high_NAs.index:
    del X_train_df[i]


columns_with_low_NAs=nas_list[nas_list<=low_threshold]

set(X_train_df['Electrical'].values)
X_train_df['Electrical']=X_train_df['Electrical'].fillna(X_train_df['Electrical'].value_counts().idxmax(), inplace=True)

set(X_train_df['MasVnrType'].values)
X_train_df['MasVnrType']=X_train_df['MasVnrType'].fillna('None')
set(X_train_df['MasVnrArea'].values)
X_train_df['MasVnrArea']=X_train_df['MasVnrArea'].fillna(0)

set(X_train_df['BsmtQual'].values)
set(X_train_df['BsmtCond'].values)
set(X_train_df['BsmtExposure'].values)
X_train_df[['BsmtCond','BsmtExposure','BsmtQual','BsmtFinType1','BsmtFinType2']]=X_train_df[['BsmtCond','BsmtExposure','BsmtQual','BsmtFinType1','BsmtFinType2']].fillna('NA')

X_train_df[['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']]=X_train_df[['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']].fillna('NA')

X_train_df.corr()['LotFrontage'].sort_values(ascending=False)

del X_train_df['LotFrontage']


sns.countplot(x=X_train_df["SaleCondition"])
sns.countplot(x=X_train_df["SaleType"])
sns.countplot(x=X_train_df["YrSold"])

X_train_df[['OverallCond','OverallQual']].corr()

sns.countplot(x=X_train_df["OverallCond"])
sns.countplot(x=X_train_df["OverallQual"])
sns.countplot(x=X_train_df["ExterCond"])
sns.countplot(x=X_train_df["ExterQual"])

sns.countplot(x=X_train_df["YrSold"])
sns.countplot(x=X_train_df["YearBuilt"])
sns.countplot(x=X_train_df["YearRemodAdd"])
sns.countplot(x=X_train_df["MoSold"])
X_train_df["YrSold"]-X_train_df["YearBuilt"]
X_train_df["YrSold"]-X_train_df["YearRemodAdd"]




#
#
#
#
