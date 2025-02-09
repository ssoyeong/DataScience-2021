import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

pd.set_option('display.max_row', 100)
pd.set_option('display.max_columns', 100)

# Read csv
df = pd.read_excel('IT_3.xlsx')
df.fillna(axis=0, method='ffill', inplace=True)

df1 = pd.DataFrame(df)
# # Drop 5 columns
df1 = df1.drop(['Miles_driven_annually', 'Years_Experience', 'EngineHP', 'credit_history'], axis=1)
# Drop rows with Nan

# Fill missing values
df1['annual_claims']=df1['annual_claims'].astype(np.int64)
df1['annual_claims']=df['annual_claims'].astype('category')






# #make target to category
# targets = df['annual_claims'].astype(np.int64)
# targets = targets.astype('category')


df_ordinal = df1.copy()
df_oneHot = df1.copy()
df_label = df1.copy()

# Convert 'Marital_Status' feature to numeric values using ordinalEncoder
ordinalEncoder = preprocessing.OrdinalEncoder()
X = pd.DataFrame(df1['Marital_Status'])
ordinalEncoder.fit(X)
df_ordinal['Marital_Status'] = pd.DataFrame(ordinalEncoder.transform(X))

# Convert 'Marital_Status' feature to numeric values using labelEncoder
labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(df_label['Marital_Status'])
df_label['Marital_Status'] = labelEncoder.transform(df_label['Marital_Status'])


# Convert 'Vehical_type' feature to numeric values using ordinalEncoder
ordEnc = preprocessing.OrdinalEncoder()
X = pd.DataFrame(df1['Vehical_type'])
ordEnc.fit(X)
df_ordinal['Vehical_type'] = pd.DataFrame(ordEnc.transform(X))

# Convert 'Vehical_type' feature to numeric values using labelEncoder
labelEnc = preprocessing.LabelEncoder()
labelEnc.fit(df1['Vehical_type'])
df_label['Vehical_type'] = pd.DataFrame(labelEnc.transform(df1['Vehical_type']))


# Convert 'Gender' features to numeric values using ordinalEncoder
X = pd.DataFrame(df1['Gender'])
ordEnc.fit(X)
df_ordinal['Gender']  = pd.DataFrame(ordEnc.transform(X))

# Convert 'Gender' features to numeric values using labelEncoder
labelEnc.fit(X)
df_label['Gender'] = pd.DataFrame(labelEnc.transform(X))


# Convert 'State' features to numeric values using ordinalEncoder
X = pd.DataFrame(df1['State'])
ordEnc.fit(X)
df_ordinal['State'] = pd.DataFrame(ordEnc.transform(X))

# Convert 'State' features to numeric values using labelEncoder
labelEnc.fit(df1['State'])
df_label['State'] = pd.DataFrame(labelEnc.transform(df1['State']))


# Convert 'Age_bucket' features to numeric values using ordinalEncoder
X = pd.DataFrame(df1['Age_bucket'])
ordEnc.fit(X)
df_ordinal['Age_bucket'] = pd.DataFrame(ordEnc.transform(X))

# Convert 'Age_bucket' features to numeric values using labelEncoder
labelEnc.fit(df1['Age_bucket'])
df_label['Age_bucket'] = pd.DataFrame(labelEnc.transform(df1['Age_bucket']))


# Convert 'EngineHP_bucket' features to numeric values using ordinalEncoder
X = pd.DataFrame(df1['EngineHP_bucket'])
ordEnc.fit(X)
df_ordinal['EngineHP_bucket'] = pd.DataFrame(ordEnc.transform(X))

# Convert 'EngineHP_bucket' features to numeric values using labelEncoder
labelEnc.fit(df1['EngineHP_bucket'])
df_label['EngineHP_bucket'] = pd.DataFrame(labelEnc.transform(df1['EngineHP_bucket']))


# Convert 'Years_Experience_bucket' features to numeric values using ordinalEncoder
X = pd.DataFrame(df1['Years_Experience_bucket'])
ordEnc.fit(X)
df_ordinal['Years_Experience_bucket'] = pd.DataFrame(ordEnc.transform(X))

# Convert 'Years_Experience_bucket' features to numeric values using labelEncoder
labelEnc.fit(df1['Years_Experience_bucket'])
df_label['Years_Experience_bucket'] = pd.DataFrame(labelEnc.transform(df1['Years_Experience_bucket']))


# Convert 'Miles_driven_annually_bucket' features to numeric values using ordinalEncoder
X = pd.DataFrame(df1['Miles_driven_annually_bucket'])
ordEnc.fit(X)
df_ordinal['Miles_driven_annually_bucket'] = pd.DataFrame(ordEnc.transform(X))

# Convert 'Miles_driven_annually_bucket' features to numeric values using labelEncoder
labelEnc.fit(df1['Miles_driven_annually_bucket'])
df_label['Miles_driven_annually_bucket'] = pd.DataFrame(labelEnc.transform(df1['Miles_driven_annually_bucket']))


# Convert 'credit_history_bucket' features to numeric values using ordinalEncoder
X = pd.DataFrame(df1['credit_history_bucket'])
ordEnc.fit(X)
df_ordinal['credit_history_bucket'] = pd.DataFrame(ordEnc.transform(X))

# Convert 'credit_history_bucket' features to numeric values using labelEncoder
labelEnc.fit(df1['credit_history_bucket'])
df_label['credit_history_bucket'] = pd.DataFrame(labelEnc.transform(df1['credit_history_bucket']))

print("=======isNan")
print(df_label.isnull().sum())
# 인코딩 거치면 null 값이 생깁니다..
df_label = df_label.dropna()

df_oneHot['annual_claims']=df_oneHot['annual_claims'].astype(np.int64)

# Getting all the categorical variables in a list
categoricalColumn = df1.columns[df1.dtypes == np.object].tolist()
# Convert categorical features to numeric values using oneHotEncoder
for col in categoricalColumn:
    if(len(df_oneHot[col].unique()) == 2):
        df_oneHot[col] = pd.get_dummies(df_oneHot[col], drop_first=True)

df_oneHot = pd.get_dummies(df_oneHot)

df1['annual_claims']=df1['annual_claims'].astype('category')
y = df1['annual_claims']
y2 = df1['annual_claims']
# Split the dataset

X1 = df_ordinal.drop(['annual_claims'], 1)
X2 = df_label.drop(['annual_claims'], 1)
X3 = df_oneHot.drop(['annual_claims'], 1)



X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, random_state=0)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, random_state=0)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y, random_state=0)



# Normalizing the ordinalEncoded dataset using MaxAbsScaler
scaler = preprocessing.MaxAbsScaler()
df_ordinal_maxAbs_train = scaler.fit_transform(X1_train)
df_ordinal_maxAbs_train = pd.DataFrame(df_ordinal_maxAbs_train, columns=X1_train.columns)

df_ordinal_maxAbs_test = scaler.fit_transform(X1_test)
df_ordinal_maxAbs_test = pd.DataFrame(df_ordinal_maxAbs_test, columns=X1_test.columns)
print(df_ordinal_maxAbs_train.head(10))



# Normalizing the oneHotEncoded dataset using MaxAbsScaler
scaler = preprocessing.MaxAbsScaler()
df_oneHot_maxAbs_train = scaler.fit_transform(X3_train)
df_oneHot_maxAbs_train = pd.DataFrame(df_oneHot_maxAbs_train, columns=X3_train.columns)

df_oneHot_maxAbs_test = scaler.fit_transform(X3_test)
df_oneHot_maxAbs_test = pd.DataFrame(df_oneHot_maxAbs_test, columns=X3_test.columns)
print(df_oneHot_maxAbs_test.head(10))



# Normalizing the labelEncoded dataset using MaxAbsScaler
scaler = preprocessing.MaxAbsScaler()
df_label_maxAbs_train = scaler.fit_transform(X2_train)
df_label_maxAbs_train = pd.DataFrame(df_label_maxAbs_train, columns=X2_train.columns)

df_label_maxAbs_test = scaler.fit_transform(X2_test)
df_label_maxAbs_test = pd.DataFrame(df_label_maxAbs_test, columns=X2_test.columns)
print(df_label_maxAbs_test.head(10))


# Normalizing the ordinalEncoded dataset using RobustScaler
scaler = preprocessing.RobustScaler()
df_ordinal_robust_train = scaler.fit_transform(X1_train)
df_ordinal_robust_train = pd.DataFrame(df_ordinal_robust_train, columns=X1_train.columns)

df_ordinal_robust_test = scaler.fit_transform(X1_test)
df_ordinal_robust_test = pd.DataFrame(df_ordinal_robust_test, columns=X1_test.columns)
print(df_ordinal_robust_test.head(10))



# Normalizing the oneHotEncoded dataset using RobustScaler
scaler = preprocessing.RobustScaler()
df_oneHot_robust_train = scaler.fit_transform(X3_train)
df_oneHot_robust_train = pd.DataFrame(df_oneHot_robust_train, columns=X3_train.columns)

df_oneHot_robust_test = scaler.fit_transform(X3_test)
df_oneHot_robust_test = pd.DataFrame(df_oneHot_robust_test, columns=X3_test.columns)
print(df_oneHot_robust_test.head(10))



# Normalizing the labelEncoded dataset using RobustScaler
scaler = preprocessing.RobustScaler()
df_label_robust_train = scaler.fit_transform(X2_train)
df_label_robust_train = pd.DataFrame(df_label_robust_train, columns=X2_train.columns)

df_label_robust_test = scaler.fit_transform(X2_test)
df_label_robust_test = pd.DataFrame(df_label_robust_test, columns=X2_test.columns)
print(df_label_robust_test.head(10))



# Normalizing the ordinalEncoded dataset using MinMaxScaler
scaler = preprocessing.MinMaxScaler()
df_ordinal_minMax_train = scaler.fit_transform(X1_train)
df_ordinal_minMax_train = pd.DataFrame(df_ordinal_minMax_train, columns=X1_train.columns)


df_ordinal_minMax_test = scaler.fit_transform(X1_test)
df_ordinal_minMax_test = pd.DataFrame(df_ordinal_minMax_test, columns=X1_test.columns)
print(df_ordinal_minMax_test.head(10))

# Normalizing the oneHotEncoded dataset using MinMaxScaler
scaler = preprocessing.MinMaxScaler()
df_oneHot_minMax_train = scaler.fit_transform(X3_train)
df_oneHot_minMax_train = pd.DataFrame(df_oneHot_minMax_train, columns=X3_train.columns)

df_oneHot_minMax_test = scaler.fit_transform(X3_test)
df_oneHot_minMax_test = pd.DataFrame(df_oneHot_minMax_test, columns=X3_test.columns)
print(df_oneHot_minMax_test.head(10))

# Normalizing the labelEncoded dataset using MinMaxScaler
scaler = preprocessing.MinMaxScaler()
df_label_minMax_train = scaler.fit_transform(X2_train)
df_label_minMax_train = pd.DataFrame(df_label_minMax_train, columns=X2_train.columns)

df_label_minMax_test = scaler.fit_transform(X2_test)
df_label_minMax_test = pd.DataFrame(df_label_minMax_test, columns=X2_test.columns)
print(df_label_minMax_test.head(10))

# Normalizing the ordinalEncoded dataset using StandardScaler
scaler = preprocessing.StandardScaler()
df_ordinal_stand_train = scaler.fit_transform(X1_train)
df_ordinal_stand_train = pd.DataFrame(df_ordinal_stand_train, columns=X1_train.columns)

df_ordinal_stand_test = scaler.fit_transform(X1_test)
df_ordinal_stand_test = pd.DataFrame(df_ordinal_stand_test, columns=X1_test.columns)
print(df_ordinal_stand_test.head(10))

# Normalizing the oneHotEncoded dataset using StandardScaler
scaler = preprocessing.StandardScaler()
df_oneHot_stand_train = scaler.fit_transform(X3_train)
df_oneHot_stand_train = pd.DataFrame(df_oneHot_stand_train, columns=X3_train.columns)

df_oneHot_stand_test = scaler.fit_transform(X3_test)
df_oneHot_stand_test = pd.DataFrame(df_oneHot_stand_test, columns=X3_test.columns)
print(df_oneHot_stand_test.head(10))

# Normalizing the labelEncoded dataset using StandardScaler
scaler = preprocessing.StandardScaler()
df_label_stand_train = scaler.fit_transform(X2_train)
df_label_stand_train = pd.DataFrame(df_label_stand_train, columns=X2_train.columns)

df_label_stand_test = scaler.fit_transform(X2_test)
df_label_stand_test = pd.DataFrame(df_label_stand_test, columns=X2_test.columns)
print(df_label_stand_test.head(10))




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

df['annual_claims']=df1['annual_claims'].astype(np.int64)
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

# Correlation of all features
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(method='pearson'), annot=True, fmt='.2f', linewidths=5, cmap='Blues')
plt.title("Correlation of all features")
plt.show()



# # Rename 'target' and 'annual_claims' features
# df2.rename(columns = {'target':'claim_prediction', 'annual_claims':'target'}, inplace=True)

# Drop 'ID' feature


X_train = df_label_stand_train
X_test = df_label_stand_test
y_train = y2_train
y_test = y2_test

print(X_test)
print(X_train)
#



"""
model = BaggingClassifier() #grid search 해야함
params = {'n_estimators': [100,125,150],
              'max_features': [0.1,0.4, 0.5,1],
              'max_samples':[0.1, 0.2, 0.3,0.5,1]
          };

print("\n---------- Bagging classifier grid search ----------")
model_gscv = GridSearchCV(model,param_grid = params,cv=5,scoring='accuracy')
model_gscv.fit(X_train,y_train)
print("Best param : ",model_gscv.best_params_)
print("Best score : ",model_gscv.best_score_)
prediction = model_gscv.predict(X_test)
print(model_gscv.score(X_test,y_test))


# Using KNN algorithm
# Create and train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_prec = knn.predict(X_test)
print("\n---------- KNN classifier  ----------")
print(y_prec[0:100])
print("Score: %.2f" % knn.score(X_test, y_test))


rfModel = RandomForestClassifier()
params = {'n_estimators': [100,125,150],
             'max_depth': [2,4,6,8],
               'max_features': [0.1,0.4, 0.5,1],
               'max_samples':[0.1, 0.2, 0.3,0.5,1]
          };
print("\n----------Random Forest classifier grid search ----------")
rfModel_gscv = GridSearchCV(rfModel,params,scoring = 'r2')
rfModel_gscv.fit(X_train,y_train)
y_predict = rfModel_gscv.predict(X_test)
print("Best param : ",rfModel_gscv.best_params_)
print("Best score : ",rfModel_gscv.best_score_)
print(rfModel_gscv.score(X_test,y_test))
print("\n\n\n")

"""
y_test = y_test.astype(np.int64)
y_train = y_train.astype(np.int)
"""
models = BaggingRegressor()
params = {'n_estimators': [100,125,150],
              'max_features': [0.1,0.4, 0.5,1],
              'max_samples':[0.1, 0.2, 0.3,0.5,1]
          };

print("\n---------- Bagging regressor grid search ----------")
models_gscv = GridSearchCV(models,param_grid = params,cv=5,scoring='r2')
models_gscv.fit(X_train,y_train)
print("Best param : ",models_gscv.best_params_)
print("Best score : ",models_gscv.best_score_)
print(models_gscv.score(X_test,y_test))


# Using KNN algorithm
# Create and train a KNN classifier
knnR = KNeighborsRegressor(n_neighbors=5)
knnR.fit(X_train, y_train)
print("\n---------- KNN regressor----------")
print("Score: %.2f" % knnR.score(X_test, y_test))


rfModelR = RandomForestRegressor()
params = {'n_estimators': [100,125,150],
             'max_depth': [2,4,6,8],
               'max_features': [0.1,0.4, 0.5,1],
               'max_samples':[0.1, 0.2, 0.3,0.5,1]
          };
print("\n----------Random Forest regressor grid search ----------")
rfModelR_gscv = GridSearchCV(rfModelR,params,scoring = 'r2')
rfModelR_gscv.fit(X_train,y_train)
print("Best param : ",rfModelR_gscv.best_params_)
print("Best score : ",rfModelR_gscv.best_score_)
print(rfModelR_gscv.score(X_test,y_test))
print("\n\n\n")
"""
# # LinearRegression
line_reg = LinearRegression();
line_reg.fit(X_train, y_train)
y_predict = line_reg.predict(X_test)
print("\n---------- LinearRegression ----------")
print("y_predict: \n", y_predict)
print("Score: %.2f" % line_reg.score(X_test, y_test))
#
# # Polynomial Regression
poly_reg = PolynomialFeatures(degree=2)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.fit_transform(X_test)
pol_reg = LinearRegression()
pol_reg.fit(X_poly_train, y_train)
y_predict = line_reg.predict(X_test)
print("\n---------- Polynomial algorithm ----------")
print("y_predict: \n", y_predict)
print("Score: %.2f" % pol_reg.score(X_poly_test, y_test))



X = X2
y = y2

X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0)

X_train2 = np.array(X_train)
X_test2 = np.array(X_test)
y_test2 = np.array(y_test)

#kmeans cluster
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train2)
correct = 0
for i in range(len(X_test2)):
    predict_me = np.array(X_test2[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = kmeans.predict(predict_me)
    if(prediction[0]==y_test2[i]):
        correct += 1
print("Score: %.2f" % (correct/len(X_test2)))


#Make own module to predict
def process_module(df, targetName):

    # Split the dataset
    y = df[targetName]
    X = df.drop([targetName], 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


    # Normalization with 4 Scaling methods
    maxAbsScaler = preprocessing.MaxAbsScaler()
    minmaxScaler = preprocessing.MinMaxScaler()
    robustScaler = preprocessing.RobustScaler()
    standardScaler = preprocessing.StandardScaler()

    df_maxAbs_scaled_train = maxAbsScaler.fit_transform(X_train)
    df_maxAbs_scaled_train = pd.DataFrame(df_maxAbs_scaled_train, columns=X_train.columns)
    df_maxAbs_scaled_test = maxAbsScaler.fit_transform(X_test)
    df_maxAbs_scaled_test = pd.DataFrame(df_maxAbs_scaled_test, columns=X_test.columns)

    df_minMax_scaled_train = minmaxScaler.fit_transform(X_train)
    df_minMax_scaled_train = pd.DataFrame(df_minMax_scaled_train, columns=X_train.columns)
    df_minMax_scaled_test = minmaxScaler.fit_transform(X_test)
    df_minMax_scaled_test = pd.DataFrame(df_minMax_scaled_test, columns=X_test.columns)

    df_robust_scaled_train = robustScaler.fit_transform(X_train)
    df_robust_scaled_train = pd.DataFrame(df_robust_scaled_train, columns=X_train.columns)
    df_robust_scaled_test = robustScaler.fit_transform(X_test)
    df_robust_scaled_test = pd.DataFrame(df_robust_scaled_test, columns=X_test.columns)

    df_standard_scaled_train = standardScaler.fit_transform(X_train)
    df_standard_scaled_train = pd.DataFrame(df_standard_scaled_train, columns=X_train.columns)
    df_standard_scaled_test = standardScaler.fit_transform(X_test)
    df_standard_scaled_test = pd.DataFrame(df_standard_scaled_test, columns=X_test.columns)


    # Alogirthm
    print("\n------------------------- Using maxAbs scaled dataset -------------------------")
    max_score_maxAbs = algorithm_module(df_maxAbs_scaled_train, df_maxAbs_scaled_test, y_train, y_test)
    print("\n------------------------- Using minMax scaled dataset -------------------------")
    max_score_minMax = algorithm_module(df_minMax_scaled_train, df_minMax_scaled_test, y_train, y_test)
    print("\n------------------------- Using robust scaled dataset -------------------------")
    max_score_robust = algorithm_module(df_robust_scaled_train, df_robust_scaled_test, y_train, y_test)
    print("\n------------------------- Using standard scaled dataset -------------------------")
    max_score_standard = algorithm_module(df_standard_scaled_train, df_standard_scaled_test, y_train, y_test)


    # Result
    max_score_result = max(max_score_maxAbs, max_score_minMax, max_score_robust, max_score_standard)
    print("\n\n============================== Result ==============================")
    print("Final maximum score: %.6f" % max_score_result)


def algorithm_module(X_train, X_test, y_train, y_test):

    # Linear Regression
    line_reg = LinearRegression()
    line_reg.fit(X_train, y_train)
    y_prec_linear = line_reg.predict(X_test)
    score_linear = line_reg.score(X_test, y_test)
    print("\ny_predict_linear: \n", y_prec_linear[0:50])
    print("Score: %.6f" % score_linear)

    # Polynomial Regression
    poly_reg = PolynomialFeatures(degree=2)
    X_poly_train = poly_reg.fit_transform(X_train)
    X_poly_test = poly_reg.fit_transform(X_test)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly_train, y_train)
    y_prec_poly = line_reg.predict(X_test)
    score_poly = pol_reg.score(X_poly_test, y_test)
    print("\ny_predict_poly: \n", y_prec_poly[0:50])
    print("Score: %.6f" % score_poly)

    # KNN algorithm
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_prec_knn = knn.predict(X_test)
    score_knn = knn.score(X_test, y_test)
    print("\ny_predict_KNN: \n", y_prec_knn[0:50])
    print("Score: %.6f" % score_knn)

    # Random Forest
    random_forest = RandomForestRegressor(max_depth=4, random_state=0)
    random_forest.fit(X_train, y_train)
    y_predict_rf = random_forest.predict(X_test)
    score_rf = random_forest.score(X_test, y_test)
    print("\ny_predict_RF: \n", y_predict_rf[0:50])
    print("Score: %.6f" % score_rf)

    max_score = max(score_linear, score_poly, score_knn, score_rf)
    return max_score


# Test our model using the ordinal encoded dataset
df_test_model = df_ordinal.copy()
print("\n\n============================== Using own module ==============================")
process_module(df_test_model, 'annual_claims')
# process_module(df_test_model, 'target')