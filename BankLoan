#IMPORTING PACKAGES

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew,mode

# LOADING THE DATA INTO WORKSPACE
# PLEASE CHOOSE THE WORKING DIRECTORY WHERE OUR DATA SET IS PRESENT

train=pd.read_csv("XYZCorp_LendingData.txt",sep="\t", encoding="latin-1")

# BACK UP FILE TO REVERT ANY CHANGES

copy_data= train.copy(deep=True)


# DATA SET WITH ONLY CONSIDERED VARIABLES BASED ON DOMAIN KNOWLEDGE
       
dataSet = copy_data[['annual_inc','int_rate' , 'collection_recovery_fee', 'last_pymnt_amnt' ,
                     'dti','loan_amnt' ,'funded_amnt','out_prncp' ,
                     'funded_amnt_inv','revol_bal' , 'installment','revol_util'
                     ,'total_pymnt','delinq_2yrs','issue_d','earliest_cr_line',
                     'last_credit_pull_d','emp_length' , 'last_pymnt_d', 'home_ownership',
                     'next_pymnt_d' ,'initial_list_status','open_acc' , 'inq_last_6mths',
                     'sub_grade' , 'pub_rec','term','verification_status']]

# INFORMATION ABOUT THE DATA

dataSet.info()
summary = dataSet.describe()

# CHECKING FOR MISSING VALUES

dataSet.isnull().sum()

# MISSING VALUE IMPUTATION

#Revol_Util Column
(dataSet['revol_util']).isnull().sum()
(dataSet['revol_util']) = np.where((dataSet['revol_util']).isnull(), ((dataSet['revol_util'])).median() , (dataSet['revol_util']))
(dataSet['revol_util']).isnull().sum()

#last_credit_pull_d Column
(dataSet['last_credit_pull_d']).isnull().sum()
(dataSet['last_credit_pull_d']).value_counts()
dataSet['last_credit_pull_d']=np.where(dataSet['last_credit_pull_d'].isnull(),dataSet['last_credit_pull_d'].mode(),dataSet['last_credit_pull_d'])
(dataSet['last_credit_pull_d']).isnull().sum()

#emp_length Column 
(dataSet['emp_length']).isnull().sum()
(dataSet['emp_length']).value_counts()
dataSet['emp_length']=np.where(dataSet['emp_length'].isnull(),dataSet['emp_length'].mode(),dataSet['emp_length'])
dataSet['emp_length'] = dataSet['emp_length'].replace('10+ years' , '10 years')
dataSet['emp_length'] = dataSet['emp_length'].replace('< 1 year' , '0 year')
(dataSet['emp_length']).isnull().sum()

#last_pymnt_d Column 
(dataSet['last_pymnt_d']).isnull().sum()
(dataSet['last_pymnt_d']).value_counts()
dataSet['last_pymnt_d']=np.where(dataSet['last_pymnt_d'].isnull(),dataSet['last_pymnt_d'].mode(),dataSet['last_pymnt_d'])
(dataSet['last_pymnt_d']).isnull().sum()

#next_pymnt_d Column
(dataSet['next_pymnt_d']).isnull().sum()
(dataSet['next_pymnt_d']).value_counts()
dataSet['next_pymnt_d']=np.where(dataSet['next_pymnt_d'].isnull(),dataSet['next_pymnt_d'].mode(),dataSet['next_pymnt_d'])
(dataSet['next_pymnt_d']).isnull().sum()

#verification_status Column
(dataSet['verification_status']).isnull().sum()
(dataSet['verification_status']).value_counts()
dataSet['verification_status'] = dataSet['verification_status'].replace('Verified' , 'Source Verified')
(dataSet['verification_status']).value_counts()

#Home Ownership Column
dataSet['home_ownership'] = np.where(((dataSet['home_ownership'] == "MORTGAGE") | 
                            (dataSet['home_ownership'] == "RENT") 
                            | (dataSet['home_ownership'] == "OWN") 
                            | (dataSet['home_ownership'] == "OTHER")),
                            dataSet['home_ownership'], "OTHER")

# POST IMPUTING THE MISSING VALUES

dataSet.isnull().sum()

# SKEWNESS CHECK/OUTLIER DETECTION

def detect_outliers(x):
	sorted(x)
	q1, q3= np.percentile(x,[25,75])
	iqr = q3 - q1
	lower_bound = q1 - (1.5 * iqr) 
	upper_bound = q3 + (1.5 * iqr) 
	return q1,q3,lower_bound,upper_bound

#annual_inc Column
outlier_annual_inc = detect_outliers(dataSet['annual_inc'])
check_annual_inc_lower = outlier_annual_inc[2] < dataSet['annual_inc']
check_annual_inc_upper = outlier_annual_inc[3] > dataSet['annual_inc']
check_annual_inc_lower.value_counts()  # 0 values
check_annual_inc_upper.value_counts()  # 38329 Values are greater than upper bound value
dataSet['annual_inc'].skew()     # 43.612

# int_rate Column
outlier_int_rate = detect_outliers(dataSet['int_rate'])
check_int_rate_lower = outlier_int_rate[2] < dataSet['int_rate']
check_int_rate_upper = outlier_int_rate[3] > dataSet['int_rate']
check_int_rate_lower.value_counts()  # 0 values
check_int_rate_upper.value_counts()  # 5918 Values are greater than upper bound value
dataSet['int_rate'].skew()     # 0.433333

# delinq_2yrs Column
outlier_delinq_2yrs = detect_outliers(dataSet['delinq_2yrs'])
check_delinq_2yrs_lower = outlier_delinq_2yrs[2] < dataSet['delinq_2yrs']
check_delinq_2yrs_upper = outlier_delinq_2yrs[3] > dataSet['delinq_2yrs']
check_delinq_2yrs_lower.value_counts()  # 692685 values
check_delinq_2yrs_upper.value_counts()  # 855696 Values are greater than upper bound value
dataSet['delinq_2yrs'].skew()     # 5.5

# inq_last_6mths Column
outlier_inq_last_6mths = detect_outliers(dataSet['inq_last_6mths'])
check_inq_last_6mths_lower = outlier_inq_last_6mths[2] < dataSet['inq_last_6mths']
check_inq_last_6mths_upper = outlier_inq_last_6mths[3] > dataSet['inq_last_6mths']
check_inq_last_6mths_lower.value_counts()  # 0 values
check_inq_last_6mths_upper.value_counts()  # 49842 Values are greater than upper bound value
dataSet['inq_last_6mths'].skew()     # 1.683

# open_acc Column
outlier_open_acc = detect_outliers(dataSet['open_acc'])
check_open_acc_lower = outlier_open_acc[2] < dataSet['open_acc']
check_open_acc_upper = outlier_open_acc[3] > dataSet['open_acc']
check_open_acc_lower.value_counts()  # 0 values
check_open_acc_upper.value_counts()  # 33373 Values are greater than upper bound value
dataSet['open_acc'].skew()     # 1.2467
dataSet['open_acc'].value_counts()

# open_acc Column
outlier_collection_recovery_fee = detect_outliers(dataSet['collection_recovery_fee'])
check_collection_recovery_fee_lower = outlier_collection_recovery_fee[2] < dataSet['collection_recovery_fee']
check_collection_recovery_fee_upper = outlier_collection_recovery_fee[3] > dataSet['collection_recovery_fee']
check_collection_recovery_fee_lower.value_counts()  # 832934 values
check_collection_recovery_fee_upper.value_counts()  # 855696 Values are greater than upper bound value
dataSet['collection_recovery_fee'].skew()     # 28.670

# last_pymnt_amnt Column
outlier_last_pymnt_amnt = detect_outliers(dataSet['last_pymnt_amnt'])
check_last_pymnt_amnt_lower = outlier_last_pymnt_amnt[2] < dataSet['last_pymnt_amnt']
check_last_pymnt_amnt_upper = outlier_last_pymnt_amnt[3] > dataSet['last_pymnt_amnt']
check_last_pymnt_amnt_lower.value_counts()  # 0 values
check_last_pymnt_amnt_upper.value_counts()  # 155381 Values are greater than upper bound value
dataSet['last_pymnt_amnt'].skew()     # 3.4087

# dti Column
dataSet['dti']=copy_data['dti']
outlier_dti = detect_outliers(dataSet['dti'])
check_dti_lower = outlier_dti[2] < dataSet['dti']
check_dti_upper = outlier_dti[3] > dataSet['dti']
check_dti_lower.value_counts()  # 0 values
check_dti_upper.value_counts()  # 70 Values are greater than upper bound value
dataSet['dti'].skew()     # 439.573

# loan_amnt Column
outlier_loan_amnt = detect_outliers(dataSet['loan_amnt'])
check_loan_amnt_lower = outlier_loan_amnt[2] < dataSet['loan_amnt']
check_loan_amnt_upper = outlier_loan_amnt[3] > dataSet['loan_amnt']
check_loan_amnt_lower.value_counts()  # 0 values
check_loan_amnt_upper.value_counts()  # 0 Values are greater than upper bound value
dataSet['loan_amnt'].skew()     # 0.682

# funded_amnt Column
outlier_funded_amnt = detect_outliers(dataSet['funded_amnt'])
check_funded_amnt_lower = outlier_funded_amnt[2] < dataSet['funded_amnt']
check_funded_amnt_upper = outlier_funded_amnt[3] > dataSet['funded_amnt']
check_funded_amnt_lower.value_counts()  # 0 values
check_funded_amnt_upper.value_counts()  # 0 Values are greater than upper bound value
dataSet['funded_amnt'].skew()     # 0.6846

# out_prncp Column
outlier_out_prncp = detect_outliers(dataSet['out_prncp'])
check_out_prncp_lower = outlier_out_prncp[2] < dataSet['out_prncp']
check_out_prncp_upper = outlier_out_prncp[3] > dataSet['out_prncp']
check_out_prncp_lower.value_counts()  # 0 values
check_out_prncp_upper.value_counts()  # 3661 Values are greater than upper bound value
dataSet['out_prncp'].skew()     # 0.9437

# funded_amnt_inv Column
outlier_funded_amnt_inv = detect_outliers(dataSet['funded_amnt_inv'])
check_funded_amnt_inv_lower = outlier_funded_amnt_inv[2] < dataSet['funded_amnt_inv']
check_funded_amnt_inv_upper = outlier_funded_amnt_inv[3] > dataSet['funded_amnt_inv']
check_funded_amnt_inv_lower.value_counts()  # 0 values
check_funded_amnt_inv_upper.value_counts()  # 0 Values are greater than upper bound value
dataSet['funded_amnt_inv'].skew()     # 0.6835

## Revol_util  Column
outlier_revol_util = detect_outliers(dataSet['revol_util'])
check_revol_util_lower = outlier_revol_util[2] < dataSet['revol_util']
check_revol_util_upper = outlier_revol_util[3] > dataSet['revol_util']
check_revol_util_lower.value_counts()   # 0 lower range
check_revol_util_upper.value_counts()   # 44 Values are greater than upper bound value
dataSet['revol_util'].skew() #Skew Value -.12364

## installment  Column
outlier_installment = detect_outliers(dataSet['installment'])
check_installment_lower = outlier_installment[2] < dataSet['installment']
check_installment_upper = outlier_installment[3] > dataSet['installment']
check_installment_lower.value_counts()   # 0 lower range
check_installment_upper.value_counts()   # 22410 Values are greater than upper bound value
dataSet['installment'].skew() #Skew Value .9378

## total_pymnt  Column
outlier_total_pymnt = detect_outliers(dataSet['total_pymnt'])
check_total_pymnt_lower = outlier_total_pymnt[2] < dataSet['total_pymnt']
check_total_pymnt_upper = outlier_total_pymnt[3] > dataSet['total_pymnt']
check_total_pymnt_lower.value_counts()   # 0 lower range
check_total_pymnt_upper.value_counts()   # 44450 Values are greater than upper bound value
dataSet['total_pymnt'].skew() #Skew Value 1.78

## revol_bal  Column
outlier_revol_bal = detect_outliers(dataSet['revol_bal'])
check_revol_bal_lower = outlier_revol_bal[2] < dataSet['revol_bal']
check_revol_bal_upper = outlier_revol_bal[3] > dataSet['revol_bal']
check_revol_bal_lower.value_counts()   # 0 lower range
check_revol_bal_upper.value_counts()   # 46667 Values are greater than upper bound value
dataSet['revol_bal'].skew() #Skew Value 16.21

#We are going to perform transformations for varibales having skewness > |1|
# All different kinds of transformations were performed. 
#Only the transformation that produced least value is considered. 

#Annual Column #Before :43.612     After : 1.324
dataSet['cbrt_annual_inc'] = dataSet['annual_inc'].apply(lambda x: (-1)*np.power(-x,1./3) if x<0 else np.power(x,1./3))
dataSet['cbrt_annual_inc'].skew()  ##1.3242

# collection_recovery_fee Column #Before :28.670     After : 7.59
dataSet['cbrt_collection_recovery_fee'] = dataSet['collection_recovery_fee'].apply(lambda x: (-1)*np.log(-x) if x<0 else np.log(1+x))
dataSet['cbrt_collection_recovery_fee'].skew() #7.59

#last_pymnt_amnt Column #Before :# 3.4087     After : 1.8311
dataSet['cbrt_last_pymnt_amnt'] = dataSet['last_pymnt_amnt'].apply(lambda x: (-1)*np.power(-x,1./3) if x<0 else np.power(x,1./3))
dataSet['cbrt_last_pymnt_amnt'].skew() # 1.8311

#dti Coulmn #Before :# 439.573    After : -0.571
dataSet['cbrt_dti'] = dataSet['dti'].apply(lambda x: (-1)*np.power(-x,1./3) if x<0 else np.power(x,1./3))
dataSet['cbrt_dti'].skew() #-0.571

#total_pymnt Column #Before :# 1.78    After : 0.174
dataSet['cbrt_total_pymnt'] = dataSet['total_pymnt'].apply(lambda x: (-1)*np.power(-x,1./3) if x<0 else np.power(x,1./3))
dataSet['cbrt_total_pymnt'].skew() #0.174

#revol_bal Column #Before :16.21   After : 0.792
dataSet['cbrt_revol_bal'] = dataSet['revol_bal'].apply(lambda x: (-1)*np.power(-x,1./3) if x<0 else np.power(x,1./3))
dataSet['cbrt_revol_bal'].skew()

# DATA VISUALIZATION
#HistoGrams
#eg: dataFrame['CoulmnName']

def Continous_Histogram(x):
    Histogram=x.hist(bins=40)
    return Histogram

#Normal Histograms
def Continous_Normal_Histogram(x):
    NormalHistogram=sns.distplot(x)
    return NormalHistogram

#Box Plot
def Continous_Boxplot(x):
    BoxPlot=sns.boxplot(x,orient='V')
    return BoxPlot

#Generating Graphs on a single Column For Sample
Continous_Histogram(dataSet['revol_util'])
Continous_Normal_Histogram(dataSet['revol_util'])
Continous_Boxplot(dataSet['revol_util'])

#BiVariate Analysis on Numerical Columns alone.

def Continous_Univariate_Bivariate(x):
    PairPlot=sns.pairplot(x)
    return PairPlot

# LABEL ENCODING FOR CATEGORICAL VARAIBLE
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dummyFrTransform = ['earliest_cr_line','last_credit_pull_d','emp_length','last_pymnt_d','home_ownership','next_pymnt_d','initial_list_status','sub_grade','term','verification_status']
for i in dummyFrTransform:
     dataSet[i] = le.fit_transform(dataSet[i])

# Perfoming Data Split based on given condition issue_d

dataSet['default_ind'] = train['default_ind']
dataSet['issue_d']=pd.to_datetime(dataSet['issue_d'])

#Train Split 
start_date_train='01-06-2007'
end_date_train='01-06-2015'
df1= dataSet[(dataSet['issue_d']>=start_date_train) & (dataSet['issue_d']<end_date_train)]

#Test Split
start_date_test='01-06-2015'
end_date_test='31-12-2015'
df2=dataSet[(dataSet['issue_d']>=start_date_test) & (dataSet['issue_d']<=end_date_test)]
trainData = df1
testData = df2

# train data 
X_train = trainData[['int_rate','loan_amnt','funded_amnt', 'out_prncp','funded_amnt_inv', 'installment', 'revol_util','delinq_2yrs','earliest_cr_line','last_credit_pull_d', 'emp_length', 'last_pymnt_d','home_ownership', 'next_pymnt_d', 'initial_list_status','open_acc','inq_last_6mths','sub_grade','pub_rec','term','verification_status','cbrt_annual_inc','cbrt_collection_recovery_fee','cbrt_last_pymnt_amnt','cbrt_total_pymnt','cbrt_revol_bal','cbrt_dti']]
y_train = trainData['default_ind']

# splitting within the train data
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y= train_test_split(X_train,y_train, test_size=0.2)

# TEST SPLIT
X_test = testData[['int_rate','loan_amnt','funded_amnt', 'out_prncp','funded_amnt_inv', 'installment', 'revol_util','delinq_2yrs','earliest_cr_line','last_credit_pull_d', 'emp_length', 'last_pymnt_d','home_ownership', 'next_pymnt_d', 'initial_list_status','open_acc','inq_last_6mths','sub_grade','pub_rec','term','verification_status','cbrt_annual_inc','cbrt_collection_recovery_fee','cbrt_last_pymnt_amnt','cbrt_total_pymnt','cbrt_revol_bal','cbrt_dti']]

y_test = testData['default_ind']


# MODEL BUILDING

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logi_reg = LogisticRegression()
logi_reg.fit(X_train,y_train)
preds_lr = logi_reg.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_logisticR = confusion_matrix(y_test,preds_lr)

#Support Vector Machine

from sklearn.svm import SVC
svc_rbf = SVC(kernel='rbf')
SupportVectorMachine_Model_rbf = svc_rbf.fit(X_train,y_train)
preds_svc_rbf = SupportVectorMachine_Model_rbf.predict(X_test)

cm_svm_rbf = confusion_matrix(y_test,preds_svc_rbf)


#sigmoid
from sklearn.svm import SVC
svc_sigmoid = SVC(kernel='sigmoid')
SupportVectorMachine_Model_sigmoid = svc_sigmoid.fit(X_train,y_train)
preds_svc_sigmoid = SupportVectorMachine_Model_sigmoid.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_svm_sigmoid = confusion_matrix(y_test,preds_svc_sigmoid)

#Decision Tree Classification

from sklearn.tree import DecisionTreeClassifier
classifier_Dt_Entropy = DecisionTreeClassifier(criterion = 'entropy')
classifier_Dt_Entropy.fit(X_train, y_train)
preds_dt = classifier_Dt_Entropy.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_dt = confusion_matrix(y_test,preds_dt)

#GINI 

from sklearn.tree import DecisionTreeClassifier
classifier_Dt_gini = DecisionTreeClassifier(criterion = 'gini')
classifier_Dt_gini.fit(X_train, y_train)
preds_dt_gini = classifier_Dt_gini.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_dt_gini = confusion_matrix(y_test,preds_dt_gini)


#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators=200)
randomForest_Model = rf.fit(X_train,y_train)
predictModel_rf = randomForest_Model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_randomForestR = confusion_matrix(y_test,predictModel_rf) 

#neural networks
from sklearn.neural_network import MLPClassifier
MLC= MLPClassifier(activation='relu',solver= 'adam',
                  max_iter=200, 
                  hidden_layer_sizes=(100,100,100))
model_mlc= MLC.fit(X_train,y_train)
predicted_mlc_x_train= MLC.predict(X_test)
confusing_matrix_x_train= confusion_matrix(y_test,predicted_mlc_x_train)

# Model Optimization
# cross validatin

from sklearn.model_selection import cross_val_score  
Logistic_regression= LogisticRegression()
SVC_MODEL= SVC()
Random_forest= RandomForestClassifier()
Decision_Tree= DecisionTreeClassifier()
Neural_network= MLPClassifier()

all_accuracy_Logistic= cross_val_score(Logistic_regression, X=X_train, y=y_train, cv=5)
print(all_accuracy_Logistic,all_accuracy_Logistic.mean(),all_accuracy_Logistic.std())    

all_accuracy_SVC= cross_val_score(SVC_MODEL, X=X_train, y=y_train, cv=5)
print(all_accuracy_SVC,all_accuracy_SVC.mean(),all_accuracy_SVC.std())   

all_accuracy_RF= cross_val_score(Random_forest, X=X_train, y=y_train, cv=5)
print(all_accuracy_RF,all_accuracy_RF.mean(),all_accuracy_RF.std())    
 
all_accuracy_DT= cross_val_score(Decision_Tree, X=X_train, y=y_train, cv=5)
print(all_accuracy_DT,all_accuracy_DT.mean(),all_accuracy_DT.std())    
 
all_accuracy_NN= cross_val_score(Neural_network, X=X_train, y=y_train, cv=5)
print(all_accuracy_NN,all_accuracy_NN.mean(),all_accuracy_NN.std())    
  

# grid search
from sklearn.model_selection import GridSearchCV
grid_parameter_rf= {
                     'n_estimators': [100,250,500,750,1000],
                      'criterion' : ['gini','entropy'],
                      'min_samples_split' : [2],
                      'bootstrap' : ['True','False']
                     } 
grid_srch_rf= GridSearchCV(RF,param_grid= grid_parameter_rf,scoring='accuracy',cv=5,n_jobs=1)
grid_srch_rf.fit(X_train,y_train)
best_parameter_rf= grid_srch_rf.best_params_
print(best_parameter_rf)
best_result_rf= grid_srch_rf.best_score_
print(best_result_rf)
