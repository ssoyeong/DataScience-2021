import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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

df_ordinal = df.copy()
df_oneHot = df.copy()
df_label = df.copy()

# Convert 'Marital_Status' feature to numeric values using ordinalEncoder
ordinalEncoder = preprocessing.OrdinalEncoder()
X = pd.DataFrame(df['Marital_Status'])
ordinalEncoder.fit(X)
df_ordinal['Marital_Status'] = pd.DataFrame(ordinalEncoder.transform(X))

# Convert 'Marital_Status' feature to numeric values using labelEncoder
labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(df_label['Marital_Status'])
df_label['Marital_Status'] = labelEncoder.transform(df_label['Marital_Status'])


# Convert 'Vehical_type' feature to numeric values using ordinalEncoder
ordEnc = preprocessing.OrdinalEncoder()
X = pd.DataFrame(df['Vehical_type'])
ordEnc.fit(X)
df_ordinal['Vehical_type'] = pd.DataFrame(ordEnc.transform(X))

# Convert 'Vehical_type' feature to numeric values using labelEncoder
labelEnc = preprocessing.LabelEncoder()
labelEnc.fit(df['Vehical_type'])
df_label['Vehical_type'] = pd.DataFrame(labelEnc.transform(df['Vehical_type']))


# Convert 'Gender' features to numeric values using ordinalEncoder
X = pd.DataFrame(df['Gender'])
ordEnc.fit(X)
df_ordinal['Gender']  = pd.DataFrame(ordEnc.transform(X))

# Convert 'Gender' features to numeric values using labelEncoder
labelEnc.fit(X)
df_label['Gender'] = pd.DataFrame(labelEnc.transform(X))


# Convert 'State' features to numeric values using ordinalEncoder
X = pd.DataFrame(df['State'])
ordEnc.fit(X)
df_ordinal['State'] = pd.DataFrame(ordEnc.transform(X))

# Convert 'State' features to numeric values using labelEncoder
labelEnc.fit(df['State'])
df_label['State'] = pd.DataFrame(labelEnc.transform(df['State']))


# Getting all the categorical variables in a list
categoricalColumn = df.columns[df.dtypes == np.object].tolist()
# Convert categorical features to numeric values using oneHotEncoder
for col in categoricalColumn:
    if(len(df_oneHot[col].unique()) == 2):
        df_oneHot[col] = pd.get_dummies(df_oneHot[col], drop_first=True)

df_oneHot = pd.get_dummies(df_oneHot)




# Normalizing the ordinalEncoded dataset using MaxAbsScaler
scaler = preprocessing.MaxAbsScaler()
df_ordinal_maxAbs = scaler.fit_transform(df_ordinal)
df_ordinal_maxAbs = pd.DataFrame(df_ordinal_maxAbs, columns=df_ordinal.columns)
print(df_ordinal_maxAbs.head(10))

# Normalizing the oneHotEncoded dataset using MaxAbsScaler
scaler = preprocessing.MaxAbsScaler()
df_oneHot_maxAbs = scaler.fit_transform(df_oneHot)
df_oneHot_maxAbs = pd.DataFrame(df_oneHot_maxAbs, columns=df_oneHot.columns)
print(df_oneHot_maxAbs.head(10))

# Normalizing the labelEncoded dataset using MaxAbsScaler
scaler = preprocessing.MaxAbsScaler()
df_label_maxAbs = scaler.fit_transform(df_label)
df_label_maxAbs = pd.DataFrame(df_label_maxAbs, columns=df_label.columns)
print(df_label_maxAbs.head(10))


# Normalizing the ordinalEncoded dataset using RobustScaler
scaler = preprocessing.RobustScaler()
df_ordinal_robust = scaler.fit_transform(df_ordinal)
df_ordinal_robust = pd.DataFrame(df_ordinal_maxAbs, columns=df_ordinal.columns)
print(df_ordinal_robust.head(10))

# Normalizing the oneHotEncoded dataset using RobustScaler
scaler = preprocessing.RobustScaler()
df_oneHot_robust = scaler.fit_transform(df_oneHot)
df_oneHot_robust = pd.DataFrame(df_oneHot_maxAbs, columns=df_oneHot.columns)
print(df_oneHot_robust.head(10))

# Normalizing the labelEncoded dataset using RobustScaler
scaler = preprocessing.RobustScaler()
df_label_robust = scaler.fit_transform(df_label)
df_label_robust = pd.DataFrame(df_label_maxAbs, columns=df_label.columns)
print(df_label_robust.head(10))

# Normalizing the ordinalEncoded dataset using MinMaxScaler
scaler = preprocessing.MinMaxScaler()
df_ordinal_minMax = scaler.fit_transform(df_ordinal)
df_ordinal_minMax = pd.DataFrame(df_ordinal_minMax, columns=df_ordinal.columns)
print(df_ordinal_minMax.head(10))

# Normalizing the oneHotEncoded dataset using MinMaxScaler
scaler = preprocessing.MinMaxScaler()
df_oneHot_minMax = scaler.fit_transform(df_oneHot)
df_oneHot_minMax = pd.DataFrame(df_oneHot_minMax, columns=df_oneHot.columns)
print(df_oneHot_minMax.head(10))

# Normalizing the labelEncoded dataset using MinMaxScaler
scaler = preprocessing.MinMaxScaler()
df_label_minMax = scaler.fit_transform(df_label)
df_label_minMax = pd.DataFrame(df_label_minMax, columns=df_label.columns)
print(df_label_minMax.head(10))

# Normalizing the ordinalEncoded dataset using StandardScaler
scaler = preprocessing.StandardScaler()
df_ordinal_stand = scaler.fit_transform(df_ordinal)
df_ordinal_stand = pd.DataFrame(df_ordinal_stand, columns=df_ordinal.columns)
print(df_ordinal_stand.head(10))

# Normalizing the oneHotEncoded dataset using StandardScaler
scaler = preprocessing.StandardScaler()
df_oneHot_stand = scaler.fit_transform(df_oneHot)
df_oneHot_stand = pd.DataFrame(df_oneHot_stand, columns=df_oneHot.columns)
print(df_ordinal_stand.head(10))

# Normalizing the labelEncoded dataset using StandardScaler
scaler = preprocessing.StandardScaler()
df_label_stand = scaler.fit_transform(df_label)
df_label_stand = pd.DataFrame(df_label_stand, columns=df_label.columns)
print(df_ordinal_stand.head(10))



df2 = df_label_stand.copy()
# Rename 'target' and 'annual_claims' features
df2.rename(columns = {'target':'claim_prediction', 'annual_claims':'target'}, inplace=True)
# Drop 'ID' feature
df2.drop(['ID'], 1, inplace=True)

# Split the dataset
X = df2.drop(['target'], 1)
y = df2['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Using KNN algorithm
# Create and train a KNN classifier
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

y_prec = knn.predict(X_test)
print("\n---------- KNN algorithm ----------")
print(y_prec[0:100])
print("Score: %.2f" % knn.score(X_test, y_test))


random_forest = RandomForestRegressor(max_depth= 4, random_state= 0)
random_forest.fit(X_train,y_train);
RF_y_predict = random_forest.predict(X_test);
print("\n\n--------- Random Forest Algorithm")
print(RF_y_predict[:50])
print("Score: %.2f" %random_forest.score(X_test,y_test))
