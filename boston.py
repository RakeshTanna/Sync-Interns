    # Importing Libraries
    
    # Utility Libraries
import numpy as np
import pandas as pd
    
    # Visualisation Libraries
import seaborn as sns
import matplotlib.pyplot as plt
    
    # Data Processing Libraries
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score

    # Algorithm Libraries
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
    
    # Math Library
import math
    
    # Importing Dataset
df = pd.read_csv(r"C:\Users\rakes\OneDrive\Desktop\boston-housing\train.csv")
    
    # Printing Dataset
print(df.head())


    # Data Analysis
    
    # Checking for columns and their respective datatypes
df.info()


print(df.shape)

    # Calculating the mean, minimum, deviation, maximum and other factors
print(df.describe())


fig=plt.figure()
ax=fig.add_subplot(1, 1, 1)
ax.hist(df['RM'], bins = 35)
plt.title("Average number of rooms Distribution ")
plt.xlabel("RM")
plt.ylabel("frequency")
plt.show()


fig=plt.figure()
ax=fig.add_subplot(1, 1, 1)
ax.hist(df['LSTAT'], bins = 35)
plt.title("Homeowners distribution with low class")
plt.xlabel("LSTAT")
plt.ylabel("frequency")
plt.show()

fig=plt.figure()
ax=fig.add_subplot(1, 1, 1)
ax.hist(df['PTRATIO'], bins = 35)
plt.title("Students to Teachers ratio distribution")
plt.xlabel("PTRATIO")
plt.ylabel("frequency")
plt.show()

fig=plt.figure()
ax=fig.add_subplot(1, 1, 1)
ax.hist(df['AGE'], bins = 35)
plt.title("Ages of Black Owned Portion in the town")
plt.xlabel("AGE")
plt.ylabel("BLACK")
plt.show()

N = 333
x = df.AGE
y = df.BLACK
colors = np.random.rand(N)
plt.scatter(x, y, c=colors)
plt.show()

# Call sns.heatmap to visualize missing values
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='Blues')

df = df.fillna(df.mean())
print("\n",df.isnull().sum())

df.rename(columns={'MEDV':'PRICE'}, inplace = True)

corr = df.corr()
print("\n",corr.shape)

print("\n",df.shape)

X = df.iloc[:,0:13] #independent columns
y = df.iloc[:,-1] #target column i.e price range
    
y = np.round(df['PRICE'])
#Apply SelectKBest class to extract top 5 best features
bestfeatures = SelectKBest(score_func=chi2)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
# Concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['SPECS','SCORE'] #naming the dataframe columns
print("\n",featureScores)

print("\n")
print(featureScores.nlargest(8,'SCORE')) #print 5 best features

# Train a RandomForestRegressor model
model = ExtraTreesClassifier()
model.fit(X,y)


    # Plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.15)
    
# a benchmark regressor that takes mean of training sample as predicted value
class BenchmarkRegressor:
    def __init__(self):
        pass

    def fit(self, X, y, **kwargs):
        self.mean = y.mean()

    def predict(self, X):
        return [self.mean] * len(X)

    def get_params(self, deep=False):
        return {}

bm_regr = BenchmarkRegressor()
lr_regr = LinearRegression()
dt_regr = DecisionTreeRegressor()
rf_regr = RandomForestRegressor()

    # create a list of models and evaluate each model 
models = [
        ('Benchmark', bm_regr),
        ('LR', lr_regr),
        ('Decision Tree', dt_regr),
        ('Random Forest', rf_regr)
]

print("Root Mean Square Error (RMSE) score\n")
scoring = 'neg_mean_squared_error'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    sqrt_cv_results = [math.sqrt(abs(i)) for i in cv_results]
    print("{}: {} ({})".format(name, np.mean(sqrt_cv_results), np.std(sqrt_cv_results)))
    print('Result from each iteration of cross validation:', cv_results, '\n')  

print("R-squared Value\n")
scoring = 'r2'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    print("{}: {} ({})".format(name, cv_results.mean(), cv_results.std()))
    print('Result from each iteration of cross validation:', cv_results, '\n')   

#Linear Regression
model1 = lr_regr
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)

rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))
rsquared_score = r2_score(y_test, y_pred)
print("----LINEAR REGRESSION----")  #1
print('RMSE score:', rmse_score)
print('R2 score:', rsquared_score)
sns.regplot(x=y_test, y=y_pred)
plt.xlabel('Price')
plt.title('Regression Plot')
plt.show()

print("Training Accuracy:",model1.score(X_train,y_train)*100)
print("Testing Accuracy:",model1.score(X_test,y_test)*100)
print("Model Accuracy:",r2_score(y,model1.predict(X))*100)
print("\n")

#Decision tree 
model2 = dt_regr
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)

rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))
rsquared_score = r2_score(y_test, y_pred)
print("----DECISION TREE----")
print('RMSE score:', rmse_score)
print('R2 score:', rsquared_score)

sns.regplot(x=y_test, y=y_pred)
plt.xlabel('Price')
plt.title('Decision Tree Plot')  #2
plt.show()

print("Training Accuracy:",model2.score(X_train,y_train)*100)
print("Testing Accuracy:",model2.score(X_test,y_test)*100)
print("Model Accuracy:",r2_score(y,model2.predict(X))*100)
print("\n")

#Random forest
model3 = rf_regr
model3.fit(X_train, y_train)
y_pred = model3.predict(X_test)

rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))
rsquared_score = r2_score(y_test, y_pred)
print("----RANDOM FOREST----")  #3
print('RMSE score:', rmse_score)
print('R2 score:', rsquared_score)
   
sns.regplot(x=y_test, y=y_pred)
plt.xlabel('Price')
plt.title('Random forest Plot')
plt.show()

print("Training Accuracy:",model3.score(X_train,y_train)*100)
print("Testing Accuracy:",model3.score(X_test,y_test)*100)
print("Model Accuracy:",r2_score(y,model3.predict(X))*100)


