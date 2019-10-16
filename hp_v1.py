

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
inputpath=''
train_df=pd.read_csv(os.path.join(inputpath,'train.csv'))
test_df=pd.read_csv(os.path.join(inputpath,'test.csv'))

train_df['label']='train'
test_df['label']='test'
train_test=train_df.append(test_df)

train_test=train_test.set_index('Id',drop=True)
#Check correlations between features

def correlation_heatmap(df):
    correlations = df.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show();
    
correlation_heatmap(train_test.loc[train_test['label']=='train'])

corr_matrix = train_test.loc[train_test['label']=='train'].corr().abs()

corr_threshold=0.80
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
to_drop=['GarageYrBlt','TotalBsmtSF','GarageCars','TotRmsAbvGrd']
train_test=train_test.drop(labels=to_drop, axis=1)



####Handling Missing Values
NA_obs_threshold=0.8
train_test = train_test.loc[train_test.isnull().mean(axis=1) < NA_obs_threshold]

nas_list=train_test.loc[train_test['label']=='train'].isna().sum()/len(train_test.loc[train_test['label']=='train'])
nas_list=nas_list[nas_list>0]


high_NA_threshold=0.8
medium_NA_threshold=0.3
low_NA_threshold=0.05

columns_with_high_NAs=nas_list[nas_list>=high_NA_threshold]
train_test=train_test.drop(labels=columns_with_high_NAs.index, axis=1)


columns_with_medium_NAs=nas_list[nas_list<=medium_NA_threshold]
columns_with_medium_NAs=columns_with_medium_NAs[columns_with_medium_NAs>low_NA_threshold]

train_test.loc[train_test['label']=='train'].corr()['LotFrontage'].sort_values(ascending=False)
train_test=train_test.drop(labels='LotFrontage', axis=1)
train_test[['GarageType','GarageFinish','GarageQual','GarageCond']]=train_test[['GarageType','GarageFinish','GarageQual','GarageCond']].fillna('NA')

columns_with_low_NAs=nas_list[nas_list<=low_NA_threshold]

train_test['Electrical']=train_test.loc[train_test['label']=='train']['Electrical'].fillna(train_test.loc[train_test['label']=='train']['Electrical'].value_counts().idxmax(), inplace=True)
train_test['MasVnrType']=train_test['MasVnrType'].fillna('None')
train_test['MasVnrArea']=train_test['MasVnrArea'].fillna(0)
train_test[['BsmtCond','BsmtExposure','BsmtQual','BsmtFinType1','BsmtFinType2']]=train_test[['BsmtCond','BsmtExposure','BsmtQual','BsmtFinType1','BsmtFinType2']].fillna('NA')

train_test.isna().values.sum()


#Data Exploration

sns.countplot(x=train_test.loc[train_test['label']=='train']["SaleCondition"])
sns.countplot(x=train_test.loc[train_test['label']=='train']["SaleType"])
sns.countplot(x=train_test.loc[train_test['label']=='train']["YrSold"])

train_test.loc[train_test['label']=='train'][['OverallCond','OverallQual']].corr()

sns.countplot(x=train_test.loc[train_test['label']=='train']["OverallCond"])
sns.countplot(x=train_test.loc[train_test['label']=='train']["OverallQual"])
sns.countplot(x=train_test.loc[train_test['label']=='train']["ExterCond"])
sns.countplot(x=train_test.loc[train_test['label']=='train']["ExterQual"])



#created new feature to account for age of house
train_test['HouseAge']=train_test['YrSold']-train_test['YearRemodAdd']
train_test=train_test.drop(labels=['YrSold','YearRemodAdd','YrSold','YearBuilt'], axis=1)
sns.countplot(x=train_test.loc[train_test['label']=='train']["HouseAge"])

#dropping FireplaceQu as redundant
train_test=train_test.drop(labels=['FireplaceQu'], axis=1)

#dropping GarageQual as redundant
sns.countplot(x=train_test.loc[train_test['label']=='train']["GarageCond"])
sns.countplot(x=train_test.loc[train_test['label']=='train']["GarageQual"])
sns.countplot(x=train_test.loc[train_test['label']=='train']["GarageType"])
sns.countplot(x=train_test.loc[train_test['label']=='train']["GarageFinish"])
sns.countplot(x=train_test.loc[train_test['label']=='train']["GarageArea"])

#dropping Utilities because not informative and heating because is redundant
train_test=train_test.drop(labels=['GarageQual'], axis=1)
sns.countplot(x=train_test.loc[train_test['label']=='train']["Utilities"])
sns.countplot(x=train_test.loc[train_test['label']=='train']["Heating"])
sns.countplot(x=train_test.loc[train_test['label']=='train']["HeatingQC"])
sns.countplot(x=train_test.loc[train_test['label']=='train']["CentralAir"])
train_test=train_test.drop(labels=['Utilities','Heating'], axis=1)

#dropping LotShape, LandContour, LandSlope because not informative
train_test=train_test.drop(labels=['LotShape','LandContour','LandSlope'], axis=1)

#dropping Condition2 because not informative
#TODO: binning Condition1
train_test=train_test.drop(labels=['Condition2'], axis=1)

#dropping Exterior2nd, RoofMatl because not informative
train_test=train_test.drop(labels=['Exterior2nd','RoofMatl'], axis=1)

#dropping BsmtFinSF1 and BsmtFinSF2 because redundant
train_test=train_test.drop(labels=['BsmtFinSF1','BsmtFinSF2'], axis=1)

#created two vars for n of bathrooms in the basement and above grade (both full and half)
train_test['Bathrooms']=train_test['HalfBath']+train_test['FullBath']+train_test['BsmtFullBath']+train_test['BsmtHalfBath']
train_test=train_test.drop(labels=['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'],axis=1)


#created Porch Area unique variable
train_test['PorchArea']=train_test['OpenPorchSF']+ train_test['EnclosedPorch']+ train_test['3SsnPorch'] + train_test['ScreenPorch']
train_test=train_test.drop(labels=['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'],axis=1)

#Data Exploration
sns.distplot(train_df['SalePrice'], kde=False, color="#172B4D", hist_kws={"alpha": 0.8})
plt.ylabel("Count")


mp = {'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0}
train_test['ExterCond'] = train_test['ExterCond'].map(mp)
train_test['HeatingQC'] = train_test['HeatingQC'].map(mp)
train_test['KitchenQual'] = train_test['KitchenQual'].map(mp)

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


#dummyfing
train_test.loc[:, train_test.columns != 'label']=pd.get_dummies(train_test.loc[:, train_test.columns != 'label'])
#splitting
x_train=train_test.loc[train_test['label']=='train']
x_test=train_test.loc[train_test['label']=='test']
y_train=train_test.loc[train_test['label']=='train']['SalePrice']

