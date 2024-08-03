import numpy as np
from sklearn.linear_model import Lasso
from scipy.stats import f
from scipy import stats
from statsmodels.formula.api import ols


def P_anova(True_labels, Predicted_labels, acc=1):
 
    True_mean = np.mean(True_labels)
    Predicted_mean = np.mean(Predicted_labels)

    # Total sum of squares (TSS) - Total variability in the dependent variables (y)
    TSS = np.sum((True_labels - True_mean)**2)
    # Regression sum of squares (SSR) - Variability in y explained by the model
    SSR = np.sum((Predicted_labels - Predicted_mean)**2)
    # Residual sum of squares (RSS) - Variability in y unexplained by the model
    SSE = np.sum((True_labels - Predicted_labels)**2)

    # Degrees of freedom
    n_samples = len(True_labels)
    df_total = n_samples - 1
    df_regression = acc # Number of predictors
    df_residual = df_total - df_regression - 1 #total dof number
    # Mean squares
    MSR = SSR / df_regression
    MSE = SSE / df_residual

    # F-statistic and p-value
    F_statistic = MSR / MSE
    p_value = 1 - f.cdf(F_statistic, df_regression, df_residual)
    return p_value

def create_dict(names, values, lf):
    D = {}
    Results = []
    for i in range(len(names)):
        D[names[i]] = values[i]
    
    for name in lf:
        if name in D:
            #print(f"{name} : {D[name]}")
            if abs(D[name])==0: 
                Results.append(f"{name} : L-Irrelevant") 
            else:
                Results.append(f"{name} : {D[name]}")
            
        if name not in D:
            #print(f"{name} : Irrelevant")
            Results.append(f"{name} : P-Irrelevant")
    return Results

def Conditions(model, y, X):
    # Conditions for reliable anova results
    residuals = y - model.predict(X)
    # Check normality of residuals
    _, normality_p_value = stats.normaltest(residuals)
    if normality_p_value < 0.05:
        print("Residuals are not normally distributed.")

    # Check homoscedasticity
    _, homoscedasticity_p_value = stats.levene(model.predict(X), residuals)
    if homoscedasticity_p_value < 0.05:
        print("Residuals exhibit heteroscedasticity.")
        

def Anova_Decomposition(model, X, y, preprocessor, lf, train_idx = 500):

    X_preprocessed = preprocessor.fit_transform(X)
    X = np.array(X_preprocessed)
    Conditions(model, y, X)
    X = np.array(X) 

    # ANOVA 
    One_way_intr = []
    Two_way_intr = []
    True_labels = np.array(y)[train_idx:]

   # SECOND PART - Oneway interactions 
    for i in range(X_preprocessed.shape[1]):
        # Consider splitting in train / test
        model.fit(X_preprocessed[:train_idx,i].reshape(-1,1), y[:train_idx]) 
        Predicted_labels = model.predict(X_preprocessed[train_idx:,i].reshape(-1,1))
        p_value = P_anova(True_labels, Predicted_labels)
        One_way_intr.append(p_value)

    # THIRD PART - Twoway interactions
    for i in range(X_preprocessed.shape[1]):
        temp = []
        for j in range(X_preprocessed.shape[1]):
            if i!=j:
               model.fit(X_preprocessed[:train_idx,[i, j]], y[:train_idx])
               Predicted_labels = model.predict(X_preprocessed[train_idx:,[i,j]])
               p_value = P_anova(True_labels, Predicted_labels, 2)
               temp.append(p_value)
        Two_way_intr.append(temp)

    # THIRD PART - Treshold, what we do is checing if p_value is less than alpha
    Alpha = 0.1 # This is statistical significance value not lasso regularization
    Relevant_one = [0]*X_preprocessed.shape[1]
    Relevant_two = [0]*X_preprocessed.shape[1]
    
    # Main treshold
    for i in range(len(One_way_intr)):
        #print(One_way_intr[i])
        if One_way_intr[i] <= Alpha:
            Relevant_one[i] = True
        else:
            Relevant_one[i] = False

    # Recall treshold
    for j,intrs in enumerate(Two_way_intr):
        count = 0
        for i in range(len(intrs)):
            if intrs[i] <= Alpha:
                count += 1
        if count > X_preprocessed.shape[1]//2:
            Relevant_two[j] = True
        else:
            Relevant_two[j] = False
    
    for i in range(len(Relevant_two)):
        if Relevant_two[i] == True:
            Relevant_one[i] = True

    #print(Relevant_one)
    #print(Relevant_two)

    idx = np.array(Relevant_one) == True
    X_filtered = X[:,idx]
    names = np.array(lf)[idx]
    
    # Lasso 
    lasso_model = Lasso(alpha=0.005)
    lasso_model.fit(X_filtered, y)
    lasso_coefficents = lasso_model.coef_
    Results = create_dict(names, lasso_coefficents,lf)
    return Results

