import numpy as np
import pandas as pd
from sklearn import preprocessing

pd.set_option('display.max_row', 100)
pd.set_option('display.max_columns', 100)

# Read csv
df = pd.read_excel('IT_3.xlsx')
# Drop 5 columns
df = df.drop(['Age_bucket','EngineHP_bucket','Years_Experience_bucket'
                 ,'Miles_driven_annually_bucket','credit_history_bucket'], axis=1)
# Fill missing values
df.fillna(df.mean(), inplace=True)
df.fillna(axis=0, method='ffill',inplace=True)
# print(df.isnull().sum())


#Using OrdinalEncoder
ordEnc = preprocessing.OrdinalEncoder()
X = pd.DataFrame(df['Vehical_type'])
ordEnc.fit(X)
ordVeh = pd.DataFrame(ordEnc.transform(X))
# print(ordVeh.head())
#Using OneHotEncoder
onehotEnc=preprocessing.OneHotEncoder()
onehotEnc.fit(X)
oneHotVeh = pd.DataFrame(onehotEnc.transform(X).toarray())
print(oneHotVeh.head())
#Using LabelEncoder
labelEnc = preprocessing.LabelEncoder()
labelEnc.fit(df['Vehical_type'])
labelVeh = pd.DataFrame(labelEnc.transform(df['Vehical_type']))
print(labelVeh.head())


# Convert 'Marital_Status' feature to numeric values using ordinalEncoder
df_ordinal = df.copy()
ordinalEncoder = preprocessing.OrdinalEncoder()
X = pd.DataFrame(df['Marital_Status'])
ordinalEncoder.fit(X)
df_ordinal['Marital_Status'] = pd.DataFrame(ordinalEncoder.transform(X))
print(df_ordinal.head(20))

# Convert 'Marital_Status' feature to numeric values using oneHotEncoder
df_oneHot = df.copy()
oneHotEncoder = preprocessing.OneHotEncoder()
oneHotEncoder.fit(X)
df_oneHot['Marital_Status'] = pd.DataFrame(oneHotEncoder.transform(X).toarray())
print(df_oneHot.head(20))

# Convert 'Marital_Status' feature to numeric values using labelEncoder
df_label = df.copy()
labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(df_label['Marital_Status'])
df_label['Marital_Status'] = labelEncoder.transform(df_label['Marital_Status'])
print(df_label.head(20))


# Convert 'Gender' features to numeric values using ordinalEncoder
X = pd.DataFrame(df['Gender'])
ordEnc.fit(X)
ordGen  = pd.DataFrame(ordEnc.transform(X))
print(ordGen.head(10))

# Convert 'Gender' features to numeric values using onehotEncoder
onehotEnc.fit(X)
oneHotGen = pd.DataFrame(onehotEnc.transform(X))
print(oneHotGen.head(10))

# Convert 'Gender' features to numeric values using labelEncoder
labelEnc.fit(X)
labelGen = pd.DataFrame(labelEnc.transform(X))
print(labelGen.head(10))


# Convert 'State' features to numeric values using ordinalEncoder
X = pd.DataFrame(df['State'])
ordEnc.fit(X)
ordVeh = pd.DataFrame(ordEnc.transform(X))
print(ordVeh.head(10))
# Convert 'State' features to numeric values using onehotEncorder
onehotEnc.fit(X)
oneHotVeh = pd.DataFrame(onehotEnc.transform(X).toarray())
print(oneHotVeh.head(10))
# Convert 'State' features to numeric values using labelEncoder
labelEnc.fit(df['State'])
labelVeh = pd.DataFrame(labelEnc.transform(df['State']))
print(labelVeh.head(10))

# 모든 categorical features를 encoding한 게 아니라 아직 안돌아갑니당
# # Normalizing the ordinalEncoded dataset using MaxAbsScaler
# scaler = preprocessing.MaxAbsScaler()
# df_ordinal_maxAbs = scaler.fit_transform(df_ordinal)
# df_ordinal_maxAbs = pd.DataFrame(df_ordinal_maxAbs, columns=df_ordinal.columns)
# print(df_ordinal_maxAbs.head(10))

# # Normalizing the oneHotEncoded dataset using MaxAbsScaler
# scaler = preprocessing.MaxAbsScaler()
# df_oneHot_maxAbs = scaler.fit_transform(df_oneHot)
# df_oneHot_maxAbs = pd.DataFrame(df_oneHot_maxAbs, columns=df_oneHot.columns)
# print(df_oneHot_maxAbs.head(10))

# # Normalizing the labelEncoded dataset using MaxAbsScaler
# scaler = preprocessing.MaxAbsScaler()
# df_label_maxAbs = scaler.fit_transform(df_label)
# df_label_maxAbs = pd.DataFrame(df_label_maxAbs, columns=df_label.columns)
# print(df_label_maxAbs.head(10))