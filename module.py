import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



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