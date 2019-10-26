

#Libraries
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib as mp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV

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
train_test['HeatingQC'] = train_test['HeatingQC'].map(mp)
train_test['KitchenQual'] = train_test['KitchenQual'].map(mp)


mp = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}
train_test['ExterCond'] = train_test['ExterCond'].map(mp)
train_test['ExterQual'] = train_test['ExterQual'].map(mp)
train_test['GarageQual'] = train_test['GarageQual'].map(mp)
train_test['BsmtQual'] = train_test['BsmtQual'].map(mp)
train_test['BsmtCond'] = train_test['BsmtCond'].map(mp)
train_test['FireplaceQu'] = train_test['FireplaceQu'].map(mp)
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
train_test['PoolQC'] = train_test['PoolQC'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0})


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

# Generate a histogram
plt.hist(list(train_test.loc[train_test['label']=='train']['SalePrice']))
plt.xlabel('Sale Price')
plt.title('Sale Price distribution')
plt.show()


plt.scatter(x=train_test.loc[train_test['label']=='train']['SalePrice'], y=train_test.loc[train_test['label']=='train']['LotArea'])
plt.xlabel('Sale Price')
plt.ylabel('LotArea')
plt.title('Sale Price distribution')
plt.show()

plt.scatter(x=train_test.loc[train_test['label']=='train']['SalePrice'], y=train_test.loc[train_test['label']=='train']['LotArea'])
plt.xlabel('Sale Price')
plt.ylabel('LotFrontage')
plt.title('Sale Price distribution')
plt.show()


sns.stripplot(x=train_test.loc[train_test['label']=='train']['OverallQual'], y=train_test.loc[train_test['label']=='train']['SalePrice'], data=train_test)
sns.swarmplot(x=train_test.loc[train_test['label']=='train']['SaleType'], y=train_test.loc[train_test['label']=='train']['SalePrice'], data=train_test)
sns.stripplot(x=train_test.loc[train_test['label']=='train']['HouseStyle'], y=train_test.loc[train_test['label']=='train']['SalePrice'], data=train_test)

#Feature Extraction and Creation
#Garages
train_test=train_test.drop(labels=["GarageCars","GarageType","GarageFinish"],axis=1)


#Year
years_corr_matrix=train_test[["GarageYrBlt","YearBuilt","YearRemodAdd"]].replace(to_replace='NA', value=np.nan).corr()
train_test['time_since_remodeling']=train_test['YrSold']-train_test['YearBuilt']
train_test=train_test.drop(labels=["GarageYrBlt","YearBuilt","YearRemodAdd"],axis=1)

#Pool
plt.scatter(x=train_test.loc[train_test['label']=='train']['PoolArea'], y=train_test.loc[train_test['label']=='train']['SalePrice'])
plt.xlabel('Sale Price')
plt.ylabel('Pool Area')
plt.title('Sale Price Pool Area Scatter')
plt.show()

train_test['Pool']=np.where(train_test['PoolArea']==0,0,1)
train_test=train_test.drop(labels=["PoolArea","PoolQC"],axis=1)

#Floors and Basement surface
floors_corr_matrix=train_test[["1stFlrSF","2ndFlrSF","TotalBsmtSF"]].replace(to_replace='NA', value=np.nan).corr()
floors_corr_matrix
train_test[["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF"]]
train_test=train_test.drop(labels=["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF"],axis=1)

#Number of baths
train_test['Bathrooms']=train_test["BsmtFullBath"]+train_test["BsmtHalfBath"]+train_test["FullBath"]+train_test["HalfBath"]
train_test=train_test.drop(labels=["BsmtFullBath","BsmtHalfBath","FullBath","HalfBath"],axis=1)

#Fireplaces
plt.scatter(x=train_test.loc[train_test['label']=='train']['Fireplaces'], y=train_test.loc[train_test['label']=='train']['SalePrice'])
plt.ylabel('Sale Price')
plt.xlabel('Fire Places')
plt.title('Sale Price Fire Places Scatter')
plt.show()

plt.scatter(x=train_test.loc[train_test['label']=='train']['FireplaceQu'], y=train_test.loc[train_test['label']=='train']['SalePrice'])
plt.ylabel('Sale Price')
plt.xlabel('Fire Places')
plt.title('Sale Price Fire Places Scatter')
plt.show()

train_test=train_test.drop(labels='Fireplaces',axis=1)

#SaleType,SaleCondition
sns.swarmplot(x=train_test.loc[train_test['label']=='train']['SaleType'], y=train_test.loc[train_test['label']=='train']['SalePrice'], data=train_test)
sns.swarmplot(x=train_test.loc[train_test['label']=='train']['SaleCondition'], y=train_test.loc[train_test['label']=='train']['SalePrice'], data=train_test)
mp = {'WD':'WD','New':'New','COD':'COD','ConLD':'ConLD','ConLI':'Oth','CWD':'Oth','ConLw':'Oth','Con':'Oth','Oth':'Oth'}
train_test['SaleType'] = train_test['SaleType'].map(mp)
sns.swarmplot(x=train_test.loc[train_test['label']=='train']['SaleType'], y=train_test.loc[train_test['label']=='train']['SalePrice'], data=train_test)

#Miscellaneous features and value of miscellaneous features
train_test.groupby('MiscFeature')['MiscVal'].mean()
plt.scatter(x=train_test['MiscFeature'], y=train_test['MiscVal'])
train_test=train_test.drop(labels=["MiscVal"],axis=1)


#Fence
train_test.groupby('Fence')['Fence'].count()
sns.swarmplot(x=train_test.loc[train_test['label']=='train']['Fence'], y=train_test.loc[train_test['label']=='train']['SalePrice'], data=train_test)
train_test=train_test.drop(labels=["Fence"],axis=1)

#Porch
train_test['PorchArea']=train_test['ScreenPorch']+train_test['3SsnPorch']+train_test['EnclosedPorch']+train_test['OpenPorchSF']

#WoodDeckSF
plt.hist(list(train_test.loc[train_test['label']=='train']['WoodDeckSF']))
plt.xlabel('WoodDeckSF')
plt.title('WoodDeckSF')
plt.show()

plt.scatter(x=train_test.loc[train_test['label']=='train']['WoodDeckSF'], y=train_test.loc[train_test['label']=='train']['SalePrice'])
train_test=train_test.drop(labels=["WoodDeckSF"],axis=1)


#PavedDrive,Alley,Street
train_test.groupby('Alley')['Alley'].count()
train_test.groupby('PavedDrive')['PavedDrive'].count()
train_test.groupby('Street')['Street'].count()
train_test=train_test.drop(labels=["PavedDrive","Alley"],axis=1)


#Functional, OverallQual, OverallCond
sns.swarmplot(x=train_test.loc[train_test['label']=='train']['Functional'], y=train_test.loc[train_test['label']=='train']['SalePrice'], data=train_test)
train_test['Functional'] = train_test['Functional'].map(
    {7:3,6:2,5:2,4:2,3:1,
     2:1,1:0,0:0})
sns.swarmplot(x=train_test.loc[train_test['label']=='train']['Functional'], y=train_test.loc[train_test['label']=='train']['SalePrice'], data=train_test)

sns.swarmplot(x=train_test.loc[train_test['label']=='train']['OverallQual'], y=train_test.loc[train_test['label']=='train']['SalePrice'], data=train_test)
sns.swarmplot(x=train_test.loc[train_test['label']=='train']['OverallCond'], y=train_test.loc[train_test['label']=='train']['SalePrice'], data=train_test)



qual_corr_matrix=train_test[["OverallQual","OverallCond","ExterCond","ExterQual"]].replace(to_replace='NA', value=np.nan).corr()
qual_corr_matrix

#Heating, Electrical, Utilities
sns.swarmplot(x=train_test.loc[train_test['label']=='train']['Heating'], y=train_test.loc[train_test['label']=='train']['SalePrice'], data=train_test)
train_test['Heating'] = train_test['Heating'].map({'GasA':'GasA','GasW':'GasW','Grav':'Oth','OthW':'Oth','Wall':'Oth','Floor':'Oth'})
sns.swarmplot(x=train_test.loc[train_test['label']=='train']['Heating'], y=train_test.loc[train_test['label']=='train']['SalePrice'], data=train_test)

sns.swarmplot(x=train_test.loc[train_test['label']=='train']['Electrical'], y=train_test.loc[train_test['label']=='train']['SalePrice'], data=train_test)
train_test['Electrical'] = train_test['Electrical'].map({'SBrkr':'SBrkr','FuseA':'Fuse','FuseF':'Fuse','FuseP':'Fuse','Mix':'Mix'})
sns.swarmplot(x=train_test.loc[train_test['label']=='train']['Electrical'], y=train_test.loc[train_test['label']=='train']['SalePrice'], data=train_test)

train_test.groupby('Utilities')['Utilities'].count()
train_test=train_test.drop(labels=["Utilities"],axis=1)

#Basement
bsmt_corr_matrix=train_test[["BsmtCond","BsmtQual"]].replace(to_replace='NA', value=np.nan).corr()
floors_corr_matrix
sns.swarmplot(x=train_test.loc[train_test['label']=='train']['BsmtCond'], y=train_test.loc[train_test['label']=='train']['SalePrice'], data=train_test)
sns.swarmplot(x=train_test.loc[train_test['label']=='train']['BsmtQual'], y=train_test.loc[train_test['label']=='train']['SalePrice'], data=train_test)
train_test['BsmtExposure'] = train_test['BsmtExposure'].map({'4':'2','3':'2','2':'1','1':'1','0':'0'})

train_test=train_test.drop(labels=["BsmtFinType2","BsmtFinType1"],axis=1)


#Exterior
train_test.groupby('Exterior1st')['Exterior1st'].count()
train_test['Exterior1st'] = train_test['Exterior1st'].map(
    {'AsbShng':'AsShng','AsphShn':'AsShng','BrkComm':'Brick','BrkFace':'Brick','CBlock':'Other','ImStucc':'Other','Stone':'Stone','Stucco':'Stucco',
     'CmentBd':'CmentBd','HdBoard':'HdBoard','MetalSd':'MetalSd','Other':'Other','Plywood':'Plywood','PreCast':'PreCast','VinylSd':'VinylSd','WdShing':'WdShing', 'Wd Sdng':'WdShing',})

train_test=train_test.drop(labels=["MasVnrType","RoofMatl","HouseStyle","Exterior2nd"],axis=1)


#Neighboorhood
train_test=train_test.drop(labels=["Condition2"],axis=1)

#Lot
train_test=train_test.drop(labels=["LandSlope","LotConfig","LandContour","LotShape","LotArea","LotFrontage"],axis=1)




####RANDOM FOREST CART
cols_to_enc=list(set(train_test.columns)-set(['label','SalePrice']))
train_test=pd.get_dummies(train_test, columns=cols_to_enc)

train = train_test.loc[train_test['label']=='train'].reset_index(drop=True, inplace=False)
train=train.drop(labels='label',axis=1)
test=train_test.loc[train_test['label']=='test'].reset_index(drop=True, inplace=False)
test=test.drop(labels='label',axis=1)

y_train=train['SalePrice']
X_train=train=train.drop(labels='SalePrice',axis=1)
X_test=test.copy()
X_test=X_test.drop(labels='SalePrice',axis=1)


clf = RandomForestRegressor()

param_grid={'bootstrap': [True],
 'max_depth': [100],
 'max_features': ['auto'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [10]}


grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, iid=False)
grid_search.fit(X_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


res = pd.DataFrame({'Id': test_df['Id'].values,
                    'SalePrice'   : grid_search.predict(X_test).astype(int)})
path=''
res.to_csv(os.path.join(path,'predictions.csv'), index=False)
