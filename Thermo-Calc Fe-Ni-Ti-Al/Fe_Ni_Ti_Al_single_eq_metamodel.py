'''

This program builds the metamodels using the single equilibrium result of Fe-Ni-Ti-Al alloy

If using pymoo (since multiobjective optimization), must run in a Python 3 environment:
conda activate pymoo

Else, can run in Python 2 (base)

'''


from pandas import DataFrame

# For data standardization (transformation of the data onto unit scale (mean=0 and variance=1), required in most machine learning)
from sklearn.preprocessing import StandardScaler

# For K means clustering on W(FCC_A1#1,CR)_max
from sklearn.cluster import KMeans

# For principal component analysis (PCA):
from sklearn.decomposition import PCA as sklearnPCA
import plotly.io as pio

# For K Nearest Neighbours (Regression):
from sklearn import neighbors

# For linear models (Regression):
from sklearn import linear_model

# For Polynomial regression model:
from sklearn.preprocessing import PolynomialFeatures

# For support vector regression:
from sklearn import svm

# For neural network regression:
from sklearn import neural_network

# For decision tree regression:
from sklearn import tree

# For ensemble methods:
from sklearn import ensemble

# For data standardization (transform the data so that they have a near-normally distribution with zero mean and unit variance)
from sklearn import preprocessing

# Use grid search with cross validation to select ML model parameters:
from sklearn.model_selection import train_test_split  # random split the data into "training data" and "testing data"
from sklearn.model_selection import GridSearchCV  # Exhaustive grid search with cross validation (CV)
from sklearn import metrics

# For Parallel Coordinates Plot
import plotly.graph_objects as go

# For material design optimization:
from scipy.optimize import differential_evolution
from scipy.optimize import NonlinearConstraint, Bounds  # constraints in differential_evolution() (NOT used in this program yet)

# For multiobjective optimization using pymoo:
#import autograd.numpy as anp
import numpy as np
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_decomposition
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.performance_indicator.hv import Hypervolume
import matplotlib.pyplot as plt


DOEfile = "../Thermo-Calc batch jobs/DOE_Fe_Ni_Ti_Al_single_eq.txt"

# Read DOE array from file:
# Full factorial: 3^6 cases:
DOE = [line.rstrip('\n') for line in open(DOEfile)]

wt_Ni3Ti_list = []
wt_Laves_list = []
wt_FCC_list = []

Ni_list = []  # to store the Ni content in all cases
Ti_list = []  # to store the Ti content in all cases
Al_list = []  # to store the Al content in all cases

for num, case in enumerate(DOE):   # each "case" is a string with the value '0 1 2' in string

    resultFile = "Fe-Ni-Ti-Al_single_eq_results/single_eq_" + format(num+1) + ".txt"

    resultFile = open(resultFile, 'r')
    result = resultFile.read()  # result is a string

    #print(num+1)

    if 'NI3TI' in result:
        # Remove all characters before 'NI3TI'
        result_Ni3Ti = result[result.find('NI3TI'):]  # .find() returns the index in the string
        result_Ni3Ti = result_Ni3Ti.split(' ')  # a list of string
        wt_Ni3Ti = result_Ni3Ti[36]  # wt_Ni3Ti is a string
        wt_Ni3Ti = wt_Ni3Ti.rstrip(',')  # remove the trailing ','
        wt_Ni3Ti = float(wt_Ni3Ti)  # convert the string in exponential form (e.g., "12E-01") to float
        #print('Ni3Ti = ' + format(wt_Ni3Ti))
        wt_Ni3Ti_list.append(wt_Ni3Ti)
    else:
        wt_Ni3Ti_list.append(0.0)

    if 'LAVES_PHASE_C14' in result:
        # Remove all characters before 'LAVES_PHASE_C14'
        result_Laves = result[result.find('LAVES_PHASE_C14'):]  # .find() returns the index in the string
        result_Laves = result_Laves.split(' ')  # a list of string
        wt_Laves = result_Laves[26]  # wt_Laves is a string
        wt_Laves = wt_Laves.rstrip(',')  # remove the trailing ','
        wt_Laves = float(wt_Laves)  # convert the string in exponential form (e.g., "12E-01") to float
        #print('Laves = ' + format(wt_Laves))
        wt_Laves_list.append(wt_Laves)
    else:
        wt_Laves_list.append(0.0)

    if 'FCC_A1' in result:
        # Remove all characters before 'FCC_A1'
        result_FCC = result[result.find('FCC_A1'):]  # .find() returns the index in the string
        result_FCC = result_FCC.split(' ')  # a list of string
        if 'FCC_A1#1' in result:
            wt_FCC = result_FCC[33]  # wt_FCC is a string
        else:  # 'FCC_A1' only, without '#1'
            wt_FCC = result_FCC[35]  # wt_FCC is a string
        wt_FCC = wt_FCC.rstrip(',')  # remove the trailing ','
        wt_FCC = float(wt_FCC)  # convert the string in exponential form (e.g., "12E-01") to float
        #print('FCC = ' + format(wt_FCC))
        wt_FCC_list.append(wt_FCC)
    else:
        wt_FCC_list.append(0.0)

    #print('\n')


    case = case.split(' ')  # DOE case. Now "case" is a list of strings ['0', '2', '1']

    Ni_list.append(15.0 + float(case[0]) * 1.0)
    Ti_list.append(1.0 + float(case[1]) * 1.0)
    Al_list.append(0.0 + float(case[2]) * 1.0)

#print(wt_Ni3Ti_list)
#print(wt_Laves_list)
#print(wt_FCC_list)

#print(Ni_list)
#print(Ti_list)
#print(Al_list)



# Upper and lower bounds used in plotting
Ni3Ti_upper = max(wt_Ni3Ti_list)
Ni3Ti_lower = min(wt_Ni3Ti_list)
Laves_upper = max(wt_Laves_list)
Laves_lower = min(wt_Laves_list)
FCC_upper = max(wt_FCC_list)
FCC_lower = min(wt_FCC_list)



#-------------------------Create pandas dataframe ------------------------------------------

# Create the pandas data frame for the composition and Ni3Ti weight percent
# First, create a dictionary:
Ni3Ti = {"Ni": Ni_list, "Ti": Ti_list, "Al": Al_list, "Ni3Ti": wt_Ni3Ti_list}
# Then, a pandas data frame:
df_Ni3Ti = DataFrame(Ni3Ti, columns= ["Ni", "Ti", "Al", "Ni3Ti"])
# Save the dataframe in .csv file:
df_Ni3Ti.to_csv("single_eq_Ni3Ti_metamodel/Ni3Ti_vs_composition.csv", index=None, header=True)

# Create the pandas data frame for the composition and Laves weight percent
# First, create a dictionary:
Laves = {"Ni": Ni_list, "Ti": Ti_list, "Al": Al_list, "Laves": wt_Laves_list}
# Then, a pandas data frame:
df_Laves = DataFrame(Laves, columns= ["Ni", "Ti", "Al", "Laves"])
# Save the dataframe in .csv file:
df_Laves.to_csv("single_eq_Laves_metamodel/Laves_vs_composition.csv", index=None, header=True)

# Create the pandas data frame for the composition and FCC weight percent
# First, create a dictionary:
FCC = {"Ni": Ni_list, "Ti": Ti_list, "Al": Al_list, "FCC": wt_FCC_list}
# Then, a pandas data frame:
df_FCC = DataFrame(FCC, columns= ["Ni", "Ti", "Al", "FCC"])
# Save the dataframe in .csv file:
df_FCC.to_csv("single_eq_FCC_metamodel/FCC_vs_composition.csv", index=None, header=True)


# Create the pandas data frame for the composition and Ni3Ti, Laves, and FCC weight percent
# First, create a dictionary:
Ni3Ti_Laves_FCC = {"Ni": Ni_list, "Ti": Ti_list, "Al": Al_list, "Ni3Ti": wt_Ni3Ti_list, "Laves": wt_Laves_list, "FCC": wt_FCC_list}
# Then, a pandas data frame:
df_Ni3Ti_Laves_FCC = DataFrame(Ni3Ti_Laves_FCC, columns= ["Ni", "Ti", "Al", "Ni3Ti", "Laves", "FCC"])
# Save the dataframe in .csv file:
df_Ni3Ti_Laves_FCC.to_csv("Ni3Ti_Laves_FCC_vs_composition.csv", index=None, header=True)
#------------------------------------------------------------------------------------------------------------------------------------------





##-------------------K means clustering on W(FCC_A1#1,*)------------------
#kmeans = KMeans(n_clusters=4).fit(df_element_fcc)
##print(kmeans.labels_)
##-----------------------------------------------------------------------------




##------------------------ Plot Principal Component Analysis (PCA) on df_element_fcc:--------------------------------
## Reference (use both plotly and scikit-learn): https://plot.ly/python/v3/ipython-notebooks/principal-component-analysis/#shortcut--pca-in-scikitlearn

## split data table into data X and class labels Y
#X = df_element_fcc.iloc[:,6:].values   # W(FCC_A1#1,*) at the end of solidification

## Data standardization (transformation of the data onto unit scale (mean=0 and variance=1), required in most machine learning)
#X_std = StandardScaler().fit_transform(X)

#sklearn_pca = sklearnPCA(n_components=3)
#Y_sklearn = sklearn_pca.fit_transform(X_std)

## Marker colors in PCA chart: https://community.plot.ly/t/plotly-colours-list/11730/3
## Number of colors must be the same as the number of clusters in K-means
#colors = {'0': 'rgb(255,0,0)',
          #'1': 'rgb(0,255,0)',
          #'2': 'rgb(0,0,255)',
          #'3': 'rgb(255,0,255)'}
          ##'4': 'rgb(0,255,255)'}

#data = []

#for name, col in zip(('0', '1', '2', '3'), colors.values()):
    #trace = dict(
            #type='scatter3d',
            #x=Y_sklearn[kmeans.labels_==int(name),0],
            #y=Y_sklearn[kmeans.labels_==int(name),1],
            #z=Y_sklearn[kmeans.labels_==int(name),2],
            #mode='markers',
            #name=name,
            #marker=dict(
                #color=col,
                #size=12,
                #line=dict(
                    #color='rgba(217, 217, 217, 0.14)',
                    #width=0.5),
                #opacity=0.8)
    #)
    #data.append(trace)


#layout = dict(xaxis=dict(title='PC1', showline=False),
              #yaxis=dict(title='PC2', showline=False))

#fig = dict(data=data, layout=layout)
## Use the offline mode of plotly:
#pio.write_html(fig, file="SS316L_Scheil_oxide_metamodel/PCA.html", auto_open=True)
##-----------------------------------------------------------------------------------------------







#---------------Data preparation for Machine Learning Metamodel (regression)---------------------

# Feature selection - select the variables that has strong contribution to the output
# Reference:  https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
# Reference:  https://scikit-learn.org/stable/modules/feature_selection.html
# Method: Correlation matrix shown in Heatmap

correlation_matrix = df_Ni3Ti_Laves_FCC.corr(method='spearman')  # 'spearman' for monotonic correlation, 'pearson' for linear correlation
print(correlation_matrix)
# Use a Heatmap to visualize the correlation matrix:
fig = go.Figure(data=go.Heatmap(z=correlation_matrix,
                                x=["Ni", "Ti", "Al", "Ni3Ti", "Laves", "FCC"],
                                y=["Ni", "Ti", "Al", "Ni3Ti", "Laves", "FCC"]))
# Use the offline mode of plotly:
pio.write_html(fig, file="Correlation matrix (Ni3Ti_Laves_FCC) (spearman).html", auto_open=True)

#correlation_matrix = df_Ni3Ti.corr(method='spearman')  # 'spearman' for monotonic correlation, 'pearson' for linear correlation
#print(correlation_matrix)
## Use a Heatmap to visualize the correlation matrix:
#fig = go.Figure(data=go.Heatmap(z=correlation_matrix,
                                #x=["Ni", "Ti", "Al", "Ni3Ti"],
                                #y=["Ni", "Ti", "Al", "Ni3Ti"]))
## Use the offline mode of plotly:
#pio.write_html(fig, file="single_eq_Ni3Ti_metamodel/Correlation matrix (Ni3Ti) (spearman).html", auto_open=True)

#correlation_matrix = df_Laves.corr(method='spearman')  # 'spearman' for monotonic correlation, 'pearson' for linear correlation
#print(correlation_matrix)
## Use a Heatmap to visualize the correlation matrix:
#fig = go.Figure(data=go.Heatmap(z=correlation_matrix,
                                #x=["Ni", "Ti", "Al", "Laves"],
                                #y=["Ni", "Ti", "Al", "Laves"]))
## Use the offline mode of plotly:
#pio.write_html(fig, file="single_eq_Laves_metamodel/Correlation matrix (Laves) (spearman).html", auto_open=True)

#correlation_matrix = df_FCC.corr(method='spearman')  # 'spearman' for monotonic correlation, 'pearson' for linear correlation
#print(correlation_matrix)
## Use a Heatmap to visualize the correlation matrix:
#fig = go.Figure(data=go.Heatmap(z=correlation_matrix,
                                #x=["Ni", "Ti", "Al", "FCC"],
                                #y=["Ni", "Ti", "Al", "FCC"]))
## Use the offline mode of plotly:
#pio.write_html(fig, file="single_eq_FCC_metamodel/Correlation matrix (FCC) (spearman).html", auto_open=True)



# split data table into data X and class labels Y

X_Ni3Ti = df_Ni3Ti.iloc[:,0:3].values  # alloy composition
Y_Ni3Ti = df_Ni3Ti.iloc[:,3].values    # Ni3Ti weight percent
X_Laves = df_Laves.iloc[:,0:3].values  # alloy composition
Y_Laves = df_Laves.iloc[:,3].values    # Laves weight percent

X_FCC = df_FCC.iloc[:,0:3].values  # alloy composition
Y_FCC = df_FCC.iloc[:,3].values    # FCC weight percent


#Split dataset into training and testing subsets

X_Ni3Ti_train, X_Ni3Ti_test, Y_Ni3Ti_train, Y_Ni3Ti_test = train_test_split(X_Ni3Ti, Y_Ni3Ti, test_size=0.3, random_state=0, shuffle=True)
X_Laves_train, X_Laves_test, Y_Laves_train, Y_Laves_test = train_test_split(X_Laves, Y_Laves, test_size=0.3, random_state=0, shuffle=True)
X_FCC_train, X_FCC_test, Y_FCC_train, Y_FCC_test = train_test_split(X_FCC, Y_FCC, test_size=0.3, random_state=0, shuffle=True)


# Data standardization (transform the data so that they have a near-normally distribution with zero mean and unit variance)
# Reference: "Data transformation with held out data" on https://scikit-learn.org/stable/modules/cross_validation.html
scaler = preprocessing.StandardScaler().fit(X_Ni3Ti_train)
X_Ni3Ti_train_transformed = scaler.transform(X_Ni3Ti_train)
X_Ni3Ti_test_transformed = scaler.transform(X_Ni3Ti_test)

scaler = preprocessing.StandardScaler().fit(X_Laves_train)
X_Laves_train_transformed = scaler.transform(X_Laves_train)
X_Laves_test_transformed = scaler.transform(X_Laves_test)

scaler = preprocessing.StandardScaler().fit(X_FCC_train)
X_FCC_train_transformed = scaler.transform(X_FCC_train)
X_FCC_test_transformed = scaler.transform(X_FCC_test)

#-------------------------------------------------------------------------------------------





#------------Metalmodel of composition --> Ni3Ti-----------------------

########### Ni3Ti #############

#poly = PolynomialFeatures(degree=2)
#X_Ni3Ti_train_transformed = poly.fit_transform(X_Ni3Ti_train_transformed)
#X_Ni3Ti_test_transformed = poly.fit_transform(X_Ni3Ti_test_transformed)
#metamodel_Ni3Ti = linear_model.LinearRegression()

#tuned_parameters = {'n_neighbors': [3, 4, 5, 6, 7, 8, 9],
                    #'weights': ['uniform', 'distance']}    # a dict (dictionary)
#score = "r2" #"neg_mean_squared_error"
#metamodel_Ni3Ti = GridSearchCV(neighbors.KNeighborsRegressor(), tuned_parameters, scoring=score, cv=4)

#tuned_parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    #'degree': [2, 3, 4],
                    #'gamma': ['scale', 'auto']}    # a dict (dictionary)
#score = "r2" #"neg_mean_squared_error"
#metamodel_Ni3Ti = GridSearchCV(svm.SVR(), tuned_parameters, scoring=score, cv=4)

#tuned_parameters = {'max_depth': [2,3,4,5,6]}    # a dict (dictionary)
#score = "r2" #"neg_mean_squared_error"
#metamodel_Ni3Ti = GridSearchCV(tree.DecisionTreeRegressor(), tuned_parameters, scoring=score, cv=4)

#tuned_parameters = {'n_estimators': [100,200,300,400]}    # a dict (dictionary)
#score = "r2" #"neg_mean_squared_error"
#metamodel_Ni3Ti = GridSearchCV(ensemble.AdaBoostRegressor(), tuned_parameters, scoring=score, cv=4)

# RandomForestRegressor is has highest R2 score for Ni3Ti metamodel
tuned_parameters = {'max_depth': [2,3,4,5,6]}    # a dict (dictionary)
score = "r2" #"neg_mean_squared_error"
metamodel_Ni3Ti = GridSearchCV(ensemble.RandomForestRegressor(), tuned_parameters, scoring=score, cv=4)



############ Laves ##############

#poly = PolynomialFeatures(degree=2)
#X_Laves_train_transformed = poly.fit_transform(X_Laves_train_transformed)
#X_Laves_test_transformed = poly.fit_transform(X_Laves_test_transformed)
#metamodel_Laves = linear_model.LinearRegression()

# KNeighborsRegressor can result in zero Laves in optimized composition, although the metamodel does not have the highest R2 score
tuned_parameters = {'n_neighbors': [3, 4, 5, 6, 7, 8, 9],
                    'weights': ['uniform', 'distance']}    # a dict (dictionary)
score = "r2" #"neg_mean_squared_error"
metamodel_Laves = GridSearchCV(neighbors.KNeighborsRegressor(), tuned_parameters, scoring=score, cv=4)

#tuned_parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    #'degree': [2, 3, 4],
                    #'gamma': ['scale', 'auto']}    # a dict (dictionary)
#score = "r2" #"neg_mean_squared_error"
#metamodel_Laves = GridSearchCV(svm.SVR(), tuned_parameters, scoring=score, cv=4)

#tuned_parameters = {'max_depth': [2,3,4,5,6]}    # a dict (dictionary)
#score = "r2" #"neg_mean_squared_error"
#metamodel_Laves = GridSearchCV(tree.DecisionTreeRegressor(), tuned_parameters, scoring=score, cv=4)

#tuned_parameters = {'n_estimators': [100,200,300,400]}    # a dict (dictionary)
#score = "r2" #"neg_mean_squared_error"
#metamodel_Laves = GridSearchCV(ensemble.AdaBoostRegressor(), tuned_parameters, scoring=score, cv=4)

## RandomForestRegressor is has highest R2 score for Laves metamodel
#tuned_parameters = {'max_depth': [2,3,4,5,6]}    # a dict (dictionary)
#score = "r2" #"neg_mean_squared_error"
#metamodel_Laves = GridSearchCV(ensemble.RandomForestRegressor(), tuned_parameters, scoring=score, cv=4)


############## FCC #################

#poly = PolynomialFeatures(degree=2)
#X_FCC_train_transformed = poly.fit_transform(X_FCC_train_transformed)
#X_FCC_test_transformed = poly.fit_transform(X_FCC_test_transformed)
#metamodel_FCC = linear_model.LinearRegression()

#tuned_parameters = {'n_neighbors': [3, 4, 5, 6, 7, 8, 9],
                    #'weights': ['uniform', 'distance']}    # a dict (dictionary)
#score = "r2" #"neg_mean_squared_error"
#metamodel_FCC = GridSearchCV(neighbors.KNeighborsRegressor(), tuned_parameters, scoring=score, cv=4)

#tuned_parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    #'degree': [2, 3, 4],
                    #'gamma': ['scale', 'auto']}    # a dict (dictionary)
#score = "r2" #"neg_mean_squared_error"
#metamodel_FCC = GridSearchCV(svm.SVR(), tuned_parameters, scoring=score, cv=4)

#tuned_parameters = {'max_depth': [2,3,4,5,6]}    # a dict (dictionary)
#score = "r2" #"neg_mean_squared_error"
#metamodel_FCC = GridSearchCV(tree.DecisionTreeRegressor(), tuned_parameters, scoring=score, cv=4)

#tuned_parameters = {'n_estimators': [100,200,300,400]}    # a dict (dictionary)
#score = "r2" #"neg_mean_squared_error"
#metamodel_FCC = GridSearchCV(ensemble.AdaBoostRegressor(), tuned_parameters, scoring=score, cv=4)

# RandomForestRegressor is has highest R2 score for FCC metamodel
tuned_parameters = {'max_depth': [2,3,4,5,6]}    # a dict (dictionary)
score = "r2" #"neg_mean_squared_error"
metamodel_FCC = GridSearchCV(ensemble.RandomForestRegressor(), tuned_parameters, scoring=score, cv=4)




###### To-do: Stacked Generalization (StackingRegressor) for delta_T metamodel




# Fit the regression models:

metamodel_Ni3Ti.fit(X_Ni3Ti_train_transformed, Y_Ni3Ti_train)
metamodel_Laves.fit(X_Laves_train_transformed, Y_Laves_train)
metamodel_FCC.fit(X_FCC_train_transformed, Y_FCC_train)





#------------------ Test the model using the testing dataset:---------------------------

Y_Ni3Ti_predict = metamodel_Ni3Ti.predict(X_Ni3Ti_test_transformed)
Y_Laves_predict = metamodel_Laves.predict(X_Laves_test_transformed)
Y_FCC_predict = metamodel_FCC.predict(X_FCC_test_transformed)


#print("\nMean squared error of the metamodel on Ni3Ti weight percent:")
#print(metrics.mean_squared_error(Y_Ni3Ti_test, Y_Ni3Ti_predict))

#print("\nMean squared error of the metamodel on Laves weight percent:")
#print(metrics.mean_squared_error(Y_Laves_test, Y_Laves_predict))

#print("\nMean squared error of the metamodel on FCC weight percent:")
#print(metrics.mean_squared_error(Y_FCC_test, Y_FCC_predict))


print("\nR^2 of the metamodel on Ni3Ti weight percent:")
print(metrics.r2_score(Y_Ni3Ti_test, Y_Ni3Ti_predict))

print("\nR^2 of the metamodel on Laves weight percent:")
print(metrics.r2_score(Y_Laves_test, Y_Laves_predict))

print("\nR^2 of the metamodel on FCC weight percent:")
print(metrics.r2_score(Y_FCC_test, Y_FCC_predict))



# Make the scatter plot to compare the predicted Y and the testing Y:
# Reference: https://plot.ly/python/line-and-scatter/
fig = go.Figure()
fig.add_trace(go.Scatter(x=Y_Ni3Ti_predict, y=Y_Ni3Ti_test,
                         mode='markers',
                         name='markers'))
fig.add_trace(go.Scatter(x=[Ni3Ti_lower, Ni3Ti_upper], y=[Ni3Ti_lower, Ni3Ti_upper],
                         mode='lines',
                         name='lines'))
fig.update_layout(xaxis_title="Y_Ni3Ti_predict; R^2 = "+format(metrics.r2_score(Y_Ni3Ti_test,Y_Ni3Ti_predict)),
                  yaxis_title="Y_Ni3Ti_test",
                  width = 1000,
                  height = 1000)
fig.update_xaxes(range=[Ni3Ti_lower, Ni3Ti_upper])
fig.update_yaxes(range=[Ni3Ti_lower, Ni3Ti_upper])
# Use the offline mode of plotly:
pio.write_html(fig, file="single_eq_Ni3Ti_metamodel/Metamodel accuracy (Ni3Ti).html", auto_open=True)


fig = go.Figure()
fig.add_trace(go.Scatter(x=Y_Laves_predict, y=Y_Laves_test,
                         mode='markers',
                         name='markers'))
fig.add_trace(go.Scatter(x=[Laves_lower, Laves_upper], y=[Laves_lower, Laves_upper],
                         mode='lines',
                         name='lines'))
fig.update_layout(xaxis_title="Y_Laves_predict; R^2 = "+format(metrics.r2_score(Y_Laves_test,Y_Laves_predict)),
                  yaxis_title="Y_Laves_test",
                  width = 1000,
                  height = 1000)
fig.update_xaxes(range=[Laves_lower, Laves_upper])
fig.update_yaxes(range=[Laves_lower, Laves_upper])
# Use the offline mode of plotly:
pio.write_html(fig, file="single_eq_Laves_metamodel/Metamodel accuracy (Laves).html", auto_open=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=Y_FCC_predict, y=Y_FCC_test,
                         mode='markers',
                         name='markers'))
fig.add_trace(go.Scatter(x=[FCC_lower, FCC_upper], y=[FCC_lower, FCC_upper],
                         mode='lines',
                         name='lines'))
fig.update_layout(xaxis_title="Y_FCC_predict; R^2 = "+format(metrics.r2_score(Y_FCC_test,Y_FCC_predict)),
                  yaxis_title="Y_FCC_test",
                  width = 1000,
                  height = 1000)
fig.update_xaxes(range=[FCC_lower, FCC_upper])
fig.update_yaxes(range=[FCC_lower, FCC_upper])
# Use the offline mode of plotly:
pio.write_html(fig, file="single_eq_FCC_metamodel/Metamodel accuracy (FCC).html", auto_open=True)

##-----------------------------------------------------------------------------------









#----------------Composition design optimization:  single-objective ------------

# Use differential evolution:
# Objective function:
def ObjectiveFunction(x, *args):

    # Decision variables: x = [Ni, Ti, Al]

    # Make sure wt%Al < wt%Ti:
    if x[2] >= x[1]:
        return 1000   # return a big number

        # Data standardization (transform the data so that they have a near-normally distribution with zero mean and unit variance)
    # Reference: "Data transformation with held out data" on https://scikit-learn.org/stable/modules/cross_validation.html
    x_transformed = scaler.transform([x])

    # Objective function values:

    ## Only required for Polynomial regression:
    #x_transformed = poly.fit_transform(x_transformed)

    Ni3Ti = metamodel_Ni3Ti.predict(x_transformed)
    Laves = metamodel_Laves.predict(x_transformed)
    FCC = metamodel_FCC.predict(x_transformed)

    # Make sure wt%Laves is less than a max value:
    maxLaves = 3.0
    if Laves > maxLaves:
        return 1000

    # Make sure wt%FCC is larger than a min value:
    minFCC = 10.0
    if FCC < minFCC:
        return 1000

    # Objectie function: maximize Ni3Ti, minimize Laves, and minimize FCC:
    # Minimize the positive terms and maximize the negative terms:
    obj = - 1.0 * Ni3Ti / Ni3Ti_upper \
          + 1.0 * Laves / Laves_upper \
          + 1.0 * (FCC-minFCC) / (FCC_upper-minFCC)

    return obj


# Bound constraint:  wt% Ni, Ti, Al
bounds = [(15.0, 20.0), (1.0, 15.0), (0.0, 10.0)]

# Optimization:
result = differential_evolution(ObjectiveFunction, bounds, maxiter=100)

print("\n==========Single objective optimization=========\n")
print("The minimum objective function value is: " + format(result.fun))

print("Optimized composition:")
print("Ni = " + format(result.x[0]) + "wt%")
print("Ti = " + format(result.x[1]) + "wt%")
print("Al = " + format(result.x[2]) + "wt%")

# The resultant Ni3Ti, Laves, and FCC wt% at the optimized composition:
x_transformed = scaler.transform([[result.x[0], result.x[1], result.x[2]]])

## Only required for Polynomial regression:
#x_transformed = poly.fit_transform(x_transformed)

print("wt% Ni3Ti = " + format(metamodel_Ni3Ti.predict(x_transformed)))
print("wt% Laves = " + format(metamodel_Laves.predict(x_transformed)))
print("wt% FCC = " + format(metamodel_FCC.predict(x_transformed)))
print("===================================================\n")
#----------------------------------------------------------------------------------------------------







##----------------Composition design optimization:  multi-objective -----------------------------

#class MyProblem(Problem):

    #def __init__(self):

        #super().__init__(n_var = 3,
                         #n_obj = 3,
                         #xl = np.array([15.0, 1.0, 0.0]),
                         #xu = np.array([20.0, 15.0, 10.0]),
                         #elementwise_evaluation=True)

    #def _evaluate(self, x, out, *args, **kwargs):

        #'''
        #if "elementwise_evaluation=True" defined in super().__init__:

            #x is a 2D numpy array.
            #Each row of x is an individual. Total row number is the population size defined in algorithm = NSGA2().
            #Total column number is the number of variables
            #Decision variables: x = np.array([Ni, Ti, Al],
                                             #[Ni, Ti, Al],
                                              #....)
        #else (by default, elementwise_evaluation=False):
            #x is a 1D numpy array. x = np.array([Ni, Ti, Al])
        #'''

        #x_transformed = scaler.transform([x])

        ## Output of metamodels:

        ### Only required for Polynomial regression:
        ##x_transformed = poly.fit_transform(x_transformed)

        #Ni3Ti = metamodel_Ni3Ti.predict(x_transformed)
        #Laves = metamodel_Laves.predict(x_transformed)
        #FCC = metamodel_FCC.predict(x_transformed)


        ## Objective function 1: maximize Ni3Ti
        ## Minimize the positive terms and maximize the negative terms:
        #f1 = - 1.0 * Ni3Ti / Ni3Ti_upper

        ## Objective function 2: minimize Laves:
        ## Minimize the positive terms and maximize the negative terms:
        #f2 =  1.0 * Laves / Laves_upper

        ## Objective function 3: minimize FCC:
        ## Minimize the positive terms and maximize the negative terms:
        #f3 =  1.0 * FCC / FCC_upper

        #out["F"] = np.column_stack([f1, f2, f3])


#problem = MyProblem()


#algorithm = NSGA2(pop_size = 20,
                  #n_offsprings = 20,
                  #sampling = get_sampling("real_random"),
                  #crossover = get_crossover("real_sbx", prob=0.9, eta=15),
                  #mutation = get_mutation("real_pm", eta=20),
                  #eliminate_duplicates = True)


#termination = get_termination("n_gen", 100)


#result = minimize(problem,
                  #algorithm,
                  #termination,
                  #seed=1,
                  #pf=problem.pareto_front(use_cache=False),
                  #save_history=True,
                  #verbose=True)

#print("\nThe Pareto optimal design variables:")
#print(result.X)  # design variable values
#print("\nThe Pareto optimal objective functions:")
#print(result.F)  # objective function values

## Plot the Pareto set in 3D scatter plot (3 objectives)
#plot = Scatter(title = "Objective Space")
#plot.add(result.F)  # objective function values
#plot.show()

##plot = Scatter(title = "Design Space")
##plot.add(result.X)  # design variable values
##plot.show()


## Create pandas dataframe of the Pareto optimum, and save to .csv file:
## First, create a dictionary:
#Pareto = {"Ni": result.X[:,0], "Ti": result.X[:,1], "Al": result.X[:,2], \
          #"f1": result.F[:,0], "f2": result.F[:,1], "f3": result.F[:,2]}
## Then, a pandas data frame:
#df_Pareto = DataFrame(Pareto, columns= ["Ni", "Ti", "Al", "f1", "f2", "f3"])
## Save the dataframe in .csv file:
#df_Pareto.to_csv("Pareto_optimum.csv", index=None, header=True)