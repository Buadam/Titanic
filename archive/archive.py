def survivalrate(col, value):
### input: col : list of columns. value: list of values to check
    print('Survival rate for:' + col[0] + '=' + str(value[0]) + ':')
    b=(titanic[col[0]]==value[0])
    for r in range (1,len(col)):
        print('and ' + col[r] + '=' + str(value[r]) + ':')
        b=pd.concat([b,titanic[col[r]]==value[r]],axis=1,join_axes=[b.index])
    if len(col)>1:
        b2=b.all(axis=1)
    else:
        b2=b
    return titanic['Survived'].iloc[np.where(b2)].sum()/b2.sum()


titanic_train["Age_binned"] = pd.cut(titanic_train["Age"],
                               bins=8,
                               labels=["0-10", "10-20", "20-30", "30-40",
                                       "40-50", "50-60", "60-70", "70-80"])
print(titanic_train[["Age_binned", "Survived"]].head(10))
print(titanic_train[["Pclass", "Embarked", "Fare"]].corr())

titanic_test["Age_binned"] = pd.cut(titanic_test["Age"],
                               bins=8,
                               labels=["0-10", "10-20", "20-30", "30-40",
                                       "40-50", "50-60", "60-70", "70-80"])

X_train = titanic_train.drop(["Name", "Ticket", "Cabin", "Age", "Survived"],
                             axis=1)
X_test = titanic_test.drop(["Name", "Ticket", "Cabin", "Age"],
                             axis=1)
