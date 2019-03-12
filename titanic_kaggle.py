import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# =========================== Read Data =======================================
# Read titanic_train data
titanic_train = pd.read_csv('data/train.csv', sep=',', na_values='',
                      index_col='PassengerId')
print(titanic_train.describe())
print(titanic_train.info())

# Read titanic_test data
titanic_test = pd.read_csv('data/test.csv', sep=',', na_values='',
                      index_col='PassengerId')
print(titanic_test.info())

# ================================= EDA =======================================
def survivalrate(col):
    """Function to calculate survival rates for a feature specified by col
    Input: col: column name, str
    """
    rates = []
    values = titanic_train[col].unique()
    for value in values:
        print('Survival rate for:' + col + '=' + str(value) + ':')
        rate = titanic_train["Survived"][titanic_train[col] == value].mean()
        print(rate)
        plt.bar(value, rate)
        rates = rates + rate
    return rates


# Plot survival rates for some specific features
plt.figure(1)
plt.subplot(3, 2, 1)
rates_mf = survivalrate("Sex")
plt.subplot(3, 2, 2)
rates_class = survivalrate("Pclass")
plt.subplot(3, 2, 3)
rates_embarked = survivalrate("Embarked")
plt.subplot(3, 2, 4)
rates_parch = survivalrate("Parch")
plt.subplot(3, 2, 5)
rates_sibsp = survivalrate("SibSp")
plt.subplot(3, 2, 6)
titanic_train["Age"].hist()
titanic_train["Age"][titanic_train["Survived"]==1].hist()
plt.show()


# ==============================Feature selection==============================
def featureselect(df, cols_to_dummies, cols_to_drop):
    """Feature selection for train and test data
    input:
    df: DataFrame,
    cols_to_drop: list of column names
    cols_to_dummies: list of column names to convert to dummies
    output: DataFrame after feature selection
    """
    # Bin age variables
    df["Age_binned"] = pd.cut(df["Age"], bins=8,
                                   labels=["0-10", "10-20", "20-30", "30-40",
                                           "40-50", "50-60", "60-70", "70-80"])
    # Convert dummie features to string
    df[cols_to_dummies] = df[cols_to_dummies].apply(lambda row: row.astype(str))
    # Get dummies for specific features
    dummies = pd.get_dummies(df[cols_to_dummies])
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(cols_to_dummies, axis=1)
    # Drop unnecessary features
    df = df.drop(cols_to_drop, axis=1)
    print(df.columns)
    # Replace NaNs in test set
    df["Fare"].fillna(X_test_df["Fare"].mean(), inplace=True)

    df_train = df.iloc[:891, :]
    df_test = df.iloc[891:, :]
    return df_train, df_test


cols_to_dummies = ["Sex", "Age_binned", "Embarked", "Pclass"]
cols_to_drop = ["Name", "Age", "Ticket", "Cabin"]
df = pd.concat([titanic_train, titanic_test], axis=0, sort=False)
X_train_df, X_test_df = featureselect(df, cols_to_dummies, cols_to_drop)

y_train_df = X_train_df["Survived"]
X_train_df = X_train_df.drop("Survived", axis=1)
X_test_df = X_test_df.drop("Survived", axis=1)

# Print remaining column names
X_train_df.columns
X_train_df.info()
X_test_df.columns
X_test_df.info()

X_test_df[X_test_df["Fare"].isnull()]

X_train = np.array(X_train_df)
y_train = np.array(y_train_df)
X_test = np.array(X_test_df)

# ====================Fitting Random Forest Classifier =======================
# Hyperparameter tuning by Grid Search
rf = RandomForestClassifier(max_features='auto', oob_score=True,
                            random_state=42, n_jobs=-1)
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_leaf": [1, 5, 10],
              "min_samples_split": [2, 4, 10, 12, 16],
              "n_estimators": [50, 100, 400, 700, 1000]}
gs = GridSearchCV(estimator=rf, param_grid=param_grid,
                  scoring='accuracy', cv=3, n_jobs=-1)
gs = gs.fit(X_train, y_train)

print(gs.best_score_)
print(gs.best_params_)


# =================Train best model on full training data =====================
rf = RandomForestClassifier(**gs.best_params_,
                            max_features='auto', oob_score=True,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Feature importance
features = X_train_df.columns
print(features)
feature_imp = pd.Series(rf.feature_importances_,
                        index=features).sort_values(ascending=False)
feature_imp.head()
plt.figure(2)
feature_imp.plot.bar()


# ========================Predict test data====================================
predictions = rf.predict(X_test)
submission = pd.DataFrame({"Survived": predictions.astype(int)}, index=titanic_test.index)
print(submission.head())
submission.to_csv('submission_2.csv')
