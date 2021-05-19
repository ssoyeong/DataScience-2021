import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

#show result of MaxAbs scaling EngineHP and credit history
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(6,5))
ax1.set_title('Before_scaling(EngineHp-credit history)')
sns.kdeplot(df_ordinal['EngineHP'],ax=ax1)
sns.kdeplot(df_ordinal['credit_history'],ax=ax1)

ax2.set_title('After_scaling(EngineHp-credit history)')
sns.kdeplot(df_ordinal_maxAbs['EngineHP'],ax=ax2)
sns.kdeplot(df_ordinal_maxAbs['credit_history'],ax=ax2)
plt.show()

#show result of MaxAbs scaling Years_Experience and annual_claims
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(6,5))
ax1.set_title('Before_scaling(Years_Experience-annual_claims)')
sns.kdeplot(df_ordinal['Years_Experience'],ax=ax1)
sns.kdeplot(df_ordinal['annual_claims'],ax=ax1)

ax2.set_title('After_scaling(Years_Experience-annual_claims)')
sns.kdeplot(df_ordinal_maxAbs['Years_Experience'],ax=ax2)
sns.kdeplot(df_ordinal_maxAbs['annual_claims'],ax=ax2)
plt.show()

#show result of MaxAbs scaling Gender and Marital Status
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(6,5))
ax1.set_title('Before_scaling(Gender-Marital_Status)')
sns.kdeplot(df_ordinal['Gender'],ax=ax1)
sns.kdeplot(df_ordinal['Marital_Status'],ax=ax1)

ax2.set_title('After_scaling(Gender-Marital_Status)')
sns.kdeplot(df_ordinal_maxAbs['Gender'],ax=ax2)
sns.kdeplot(df_ordinal_maxAbs['Marital_Status'],ax=ax2)
plt.show()

#show result of MaxAbs scaling Vehical_type and Miles_driven_annually 
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(6,5))
ax1.set_title('Before_scaling(Vehical_type-Miles_driven_annually)')
sns.kdeplot(df_ordinal['Vehical_type'],ax=ax1)
sns.kdeplot(df_ordinal['Miles_driven_annually'],ax=ax1)

ax2.set_title('After_scaling(Vehical_type-Miles_driven_annually)')
sns.kdeplot(df_ordinal_maxAbs['Vehical_type'],ax=ax2)
sns.kdeplot(df_ordinal_maxAbs['Miles_driven_annually'],ax=ax2)
plt.show()

#show result of MaxAbs scaling size_of_family and State 
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(6,5))
ax1.set_title('Before_scaling(size_of_family-State)')
sns.kdeplot(df_ordinal['size_of_family'],ax=ax1)
sns.kdeplot(df_ordinal['State'],ax=ax1)

ax2.set_title('After_scaling(size_of_family-State)')
sns.kdeplot(df_ordinal_maxAbs['size_of_family'],ax=ax2)
sns.kdeplot(df_ordinal_maxAbs['State'],ax=ax2)
plt.show()


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
print(df_oneHot_stand.head(10))

# Normalizing the labelEncoded dataset using StandardScaler
scaler = preprocessing.StandardScaler()
df_label_stand = scaler.fit_transform(df_label)
df_label_stand = pd.DataFrame(df_label_stand, columns=df_label.columns)
print(df_label_stand.head(10))



#bar graph of target
sns.countplot(df['target'])
plt.show()

#bar graph of Gender
sns.countplot(df['Gender'])
plt.show()

#bar graph of annual_claims
sns.countplot(df['annual_claims'])
plt.show()

#bar graph of Marital_Status
sns.countplot(df['Marital_Status'])
plt.show()

#bar graph of Vehical_type
sns.countplot(df['Vehical_type'])
#plt.show()


#bar graph of size_of_family
sns.countplot(df['size_of_family'])
#plt.show()

#bar graph of State
sns.countplot(df['State'])
plt.show()

#Two variable plots
df_f = pd.DataFrame(df[df['Gender']=='F'])
df_m = pd.DataFrame(df[df['Gender']=='M'])


#EngineHP histogram by Gender
plt.hist(df_f['EngineHP'], bins=10)
plt.title('Female')
plt.xlabel('EngineHP')
plt.ylabel('Frequency')
plt.show()

plt.hist(df_m['EngineHP'], bins=10)
plt.title('Male')
plt.xlabel('EngineHP')
plt.ylabel('Frequency')
plt.show()

#credit history histogram by Gender
plt.hist(df_m['credit_history'], bins=10)
plt.title('Male')
plt.xlabel('credit_history')
plt.ylabel('Frequency')
plt.show()

plt.hist(df_f['credit_history'], bins=10)
plt.title('Female')
plt.xlabel('credit_history')
plt.ylabel('Frequency')
plt.show()


#years exprience histogram by Gender

plt.hist(df_f['Years_Experience'], bins=10)
plt.title('Female')
plt.xlabel('Years_Experience')
plt.ylabel('Frequency')
plt.show()

plt.hist(df_m['Years_Experience'], bins=10)
plt.title('Male')
plt.xlabel('Years_Experience')
plt.ylabel('Frequency')
plt.show()

#annual claims histogram by Gender


plt.hist(df_m['annual_claims'], bins=10)
plt.title('Male')
plt.xlabel('annual_claims')
plt.ylabel('Frequency')
plt.show()

plt.hist(df_f['annual_claims'], bins=10)
plt.title('Female')
plt.xlabel('annual_claims')
plt.ylabel('Frequency')
plt.show()
#Miles driven annually histogram by Gender

plt.hist(df_f['Miles_driven_annually'], bins=10)
plt.title('Female')
plt.xlabel('Miles_driven_annually')
plt.ylabel('Frequency')
plt.show()

plt.hist(df_m['Miles_driven_annually'], bins=10)
plt.title('Male')
plt.xlabel('Miles_driven_annually')
plt.ylabel('Frequency')
plt.show()

#Size of family histogram by Gender

plt.hist(df_f['size_of_family'], bins=10)
plt.title('Female')
plt.xlabel('size_of_family')
plt.ylabel('Frequency')
plt.show()

plt.hist(df_m['size_of_family'], bins=10)
plt.title('Male')
plt.xlabel('size_of_family')
plt.ylabel('Frequency')
plt.show()

#State histogram by Gender

plt.hist(df_f['State'], bins=10)
plt.title('Female')
plt.xlabel('State')
plt.ylabel('Frequency')
plt.show()

plt.hist(df_m['State'], bins=10)
plt.title('Male')
plt.xlabel('State')
plt.ylabel('Frequency')
plt.show()


#Pie chart of vehicle type
car = pd.DataFrame(df[df['Vehical_type']=='Car'])
van = pd.DataFrame(df[df['Vehical_type']=='Van'])
truck = pd.DataFrame(df[df['Vehical_type']=='Truck'])
utility = pd.DataFrame(df[df['Vehical_type']=='Utility'])

langs = ['Car','Van','Truck','Utility']
vehicle_level = [len(car), len(van), len(truck),len(utility)]
plt.pie(vehicle_level,labels=langs,autopct='%1.2f%%')
plt.show()

#Pie chart of annual_claim
ann_0 = pd.DataFrame(df[df['annual_claims']==0])
ann_1 = pd.DataFrame(df[df['annual_claims']==1])
ann_2 = pd.DataFrame(df[df['annual_claims']==2])
ann_3 = pd.DataFrame(df[df['annual_claims']==3])
ann_4 = pd.DataFrame(df[df['annual_claims']==4])

langs = ['0','1','2','3','4']
claim_level = [len(ann_0), len(ann_1), len(ann_2),len(ann_3),len(ann_4)]
plt.pie(claim_level,labels=langs,autopct='%1.2f%%')
plt.title('annual claim levels')
plt.show()


#Pie chart of size of family
fam1 = pd.DataFrame(df[df['size_of_family']==1])
fam2 = pd.DataFrame(df[df['size_of_family']==2])
fam3 = pd.DataFrame(df[df['size_of_family']==3])
fam4 = pd.DataFrame(df[df['size_of_family']==4])
fam5 = pd.DataFrame(df[df['size_of_family']==5])
fam6 = pd.DataFrame(df[df['size_of_family']==6])
fam7 = pd.DataFrame(df[df['size_of_family']==7])
fam8 = pd.DataFrame(df[df['size_of_family']==8])


langs = ['1','2','3','4','5','6','7','8']
family_level = [len(fam1), len(fam2), len(fam3),len(fam4),len(fam5),len(fam6),len(fam7),len(fam8)]
plt.pie(family_level,labels=langs,autopct='%1.2f%%')
plt.title('famliy size levels')
plt.show()

#EngineHP and crdit history scatter plot divided by Gender

plt.scatter(df_f['EngineHP'], df_f['credit_history'],color='r')
plt.scatter(df_m['EngineHP'], df_m['credit_history'],color='b')
plt.xlabel('EngineHP')
plt.ylabel('Credit history')
plt.title('Scatter plot divided by Gender')
plt.show()

#Years_Experience and annual claims scatter plot divided by Gender

plt.scatter(df_f['Years_Experience'], df_f['annual_claims'],color='r')
plt.scatter(df_m['Years_Experience'], df_m['annual_claims'],color='b')
plt.xlabel('Years_Experience')
plt.ylabel('Annual claims ')
plt.title('Scatter plot divided by Gender')
plt.show()


#Boxplot of annual claims divided by gender
sns.boxplot(df['Gender'],df['annual_claims'])
plt.show()

#Boxplot of EngineHP divided by gender
sns.boxplot(df['Gender'],df['EngineHP'])
plt.show()


#Boxplot of Years Experience divided by gender
sns.boxplot(df['Gender'],df['Years_Experience'])
plt.show()

#Boxplot of size of family divided by gender
sns.boxplot(df['Gender'],df['size_of_family'])
plt.show()

#heatmap-pearson
sns.heatmap(df.corr(method='pearson'))
plt.title("pearson")
plt.show()




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
