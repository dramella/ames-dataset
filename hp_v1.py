

#Libraries
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

#Data input
inputpath='/Users/Debora/Desktop/kaggle-repo/'
train_df=pd.read_csv(os.path.join(inputpath,'train.csv'))
test_df=pd.read_csv(os.path.join(inputpath,'test.csv'))

train_df['label']='train'
test_df['label']='test'
train_test=train_df.append(test_df)

train_test=train_test.set_index('Id',drop=True)

####Handling Missing Values

columns_with_nas=train_test.isnull().sum()/len(train_test)

NA_obs_threshold=0.8
train_test = train_test.loc[train_test.isnull().mean(axis=1) < NA_obs_threshold]

high_NA_threshold=0.8
medium_NA_threshold=0.3
low_NA_threshold=0.05

features_with_high_NAs=columns_with_nas[columns_with_nas>=high_NA_threshold]
features_with_medium_NAs=columns_with_nas[(columns_with_nas<high_NA_threshold) & (columns_with_nas>low_NA_threshold)]
features_with_low_NAs=columns_with_nas[(columns_with_nas<=low_NA_threshold) & (columns_with_nas!=0)]

train_test[features_with_high_NAs.index]
train_test[features_with_high_NAs.index]=train_test[features_with_high_NAs.index].fillna('NA')

train_test[features_with_medium_NAs.index]
features_with_medium_NAs=features_with_medium_NAs[~features_with_medium_NAs.index.isin(list('SalePrice'))]
train_test['LotFrontage']=train_test.loc[train_test['label']=='train']['LotFrontage'].median()
train_test[features_with_medium_NAs[features_with_medium_NAs.index!='LotFrontage'].index]=train_test[features_with_medium_NAs[features_with_medium_NAs.index!='LotFrontage'].index].fillna('NA')


train_test[features_with_low_NAs.index]
train_test[['BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual']]=train_test[['BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual']].fillna('NA')
train_test[['BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','TotalBsmtSF']]=train_test[['BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','TotalBsmtSF']].fillna(0)
train_test['Electrical']=train_test['Electrical'].fillna(train_test.loc[train_test['label']=='train']['Electrical'].value_counts().idxmax())
train_test['Utilities']=train_test['Utilities'].fillna(train_test.loc[train_test['label']=='train']['Electrical'].value_counts().idxmax())

train_test['Exterior1st']=train_test['Exterior1st'].fillna(train_test.loc[train_test['label']=='train']['Exterior1st'].value_counts().idxmax())
train_test['Exterior2nd']=train_test['Exterior2nd'].fillna(train_test.loc[train_test['label']=='train']['Exterior2nd'].value_counts().idxmax())

train_test['Functional']=train_test['Functional'].fillna(train_test.loc[train_test['label']=='train']['Functional'].value_counts().idxmax())
train_test['SaleType']=train_test['SaleType'].fillna(train_test.loc[train_test['label']=='train']['SaleType'].value_counts().idxmax())

train_test[['GarageArea','GarageCars']]=train_test[['GarageArea','GarageCars']].fillna(0)

train_test['MasVnrType']=train_test['MasVnrType'].fillna('None')
train_test['MasVnrArea']=train_test['MasVnrArea'].fillna(0)

train_test['KitchenQual']=train_test['KitchenQual'].fillna(train_test.loc[train_test['label']=='train']['KitchenQual'].value_counts().idxmax())
train_test['MSZoning']=train_test['MSZoning'].fillna(train_test.loc[train_test['label']=='train']['MSZoning'].value_counts().idxmax())

columns_with_nas=train_test.isnull().sum()/len(train_test)

#Mapping of categorical variables into numerical variables

mp = {'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0}
train_test['ExterCond'] = train_test['ExterCond'].map(mp)
train_test['ExterQual'] = train_test['ExterQual'].map(mp)
train_test['GarageQual'] = train_test['GarageQual'].map(mp)
train_test['GarageCond'] = train_test['GarageCond'].map(mp)
train_test['PoolQC'] = train_test['PoolQC'].map(mp)

train_test['HeatingQC'] = train_test['HeatingQC'].map(mp)
train_test['KitchenQual'] = train_test['KitchenQual'].map(mp)
train_test['FireplaceQu'] = train_test['FireplaceQu'].map(mp)


mp = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}
train_test['BsmtQual'] = train_test['BsmtQual'].map(mp)
train_test['BsmtCond'] = train_test['BsmtCond'].map(mp)
train_test['BsmtExposure'] = train_test['BsmtExposure'].map(
    {'Gd':4,'Av':3,'Mn':2,'No':1,'NA':0})

mp = {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0}
train_test['BsmtFinType1'] = train_test['BsmtFinType1'].map(mp)
train_test['BsmtFinType2'] = train_test['BsmtFinType2'].map(mp)

train_test['CentralAir'] = train_test['CentralAir'].map({'Y':1,'N':0})
train_test['Functional'] = train_test['Functional'].map(
    {'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,
     'Maj2':2,'Sev':1,'Sal':0})

train_test['GarageFinish'] = train_test['GarageFinish'].map({'Fin':3,'RFn':2,'Unf':1,'NA':0})
train_test['GarageCond'] = train_test['GarageCond'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0})


#Data Exploration


# Create |correlation matrix|
corr_matrix = train_test.corr().abs()
corr_matrix=corr_matrix[corr_matrix.notnull()]

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

corr_thr=0.80
# Find index of feature columns with correlation greater than 0.95
highly_corr = [column for column in upper.columns if any(upper[column] > corr_thr)]

#Create a correlation matrix
corr = train_test.corr()

# plot the heatmap
sns.heatmap(corr[highly_corr])

#plot distribution of sale price
sns.distplot(list(train_test.loc[train_test['label']=='train']['SalePrice']),label='Sale Price')



