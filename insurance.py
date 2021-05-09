import numpy as np
import pandas as pd
# read csv
df = pd.read_excel('IT_3.xlsx')
df = df.drop(['Age_bucket','EngineHP_bucket','Years_Experience_bucket','Miles_driven_annually_bucket','credit_history_bucket'], axis=1)
df.fillna(df.mean(), inplace=True)
df.fillna(axis=0, method='ffill',inplace=True)

#Using OrdinalEncoder
ordEnc = preprocessing.OrdinalEncoder()
X = pd.DataFrame(df['Vehical_type'])
ordEnc.fit(X)
ordVeh = pd.DataFrame(ordEnc.transform(X))
print(ordVeh.head())
#Using OneHotEncoder
onehotEnc=preprocessing.OneHotEncoder()
onehotEnc.fit(X)
oneHotVeh = pd.DataFrame(oneHotVeh.transform(X).toarray())
print(oneHotEncoded.head())
#Using LabelEncoder
labelEnc = preprocessing.LabelEncoder()
labelEnc.fit(df['Vehical_type'])
labelVeh = pd.DataFrame(labelEnc.transform(df['Vehical_type']))
print(labelVeh.head())
