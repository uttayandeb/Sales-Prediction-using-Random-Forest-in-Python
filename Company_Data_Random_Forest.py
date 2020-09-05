
##############################################################################
####################### R A N D O M    F O R E S T ###########################
##############################################################################

#### Importing packages and loading dataset ############
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

company_data = pd.read_csv("C:\\Users\\home\Desktop\\Data Science Assignments\\Python_codes\\Random_Forest\\Company_Data.csv")



########### E D A ################

company_data.head()
company_data.columns
colnames = list(company_data.columns)


#Since our target variable is sales and it is a continuous data 
#so lets convert it into two categories(high and low) as per the median value

company_data["Sales"].unique()

company_data["Sales"].value_counts()

np.median(company_data["Sales"]) # 7.49
##therefore the middle value of sales is 7.49

company_data["sales"]="<=7.49"# creating a new column "sales" and assignning <=7.49
company_data.loc[company_data["Sales"]>=7.49,"sales"]=">=7.49"
#initializing the values as >=7.49 which are greater then 7.49

company_data.drop(["Sales"],axis=1,inplace=True)#dropping down the Sales column which is not required now


##Encoding the data as model.fit doesnt convert string data to float
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_names in company_data.columns:
    if company_data[column_names].dtype == object:
        company_data[column_names]= le.fit_transform(company_data[column_names])
    else:
        pass
#Labelling the categorical columns with 0 and 1  

  
##Splitting the data into input and output
featues = company_data.iloc[:,0:10]# i/p features
labels = company_data.iloc[:,10]# "sales" target variable

##Splitting the data into TRAIN and TEST 
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(featues,labels,test_size = 0.3,stratify = labels) 


y_train.value_counts()
#1    141
#0    139

y_test.value_counts()
#1    60
#0    60


################ MODEL BULIDING ################################

from sklearn.ensemble import RandomForestClassifier as RF

model =RF(n_jobs=4,n_estimators = 150, oob_score =True,criterion ='entropy') 
model.fit(x_train,y_train)# Fitting RandomForestClassifier model from sklearn.ensemble 
model.oob_score_
#0.7857142857142857   #### 78.6%



 
model.estimators_ # 
model.classes_ # class labels (output)
model.n_classes_ # Number of levels in class labels -2
model.n_features_  # Number of input features in model 10 here.

model.n_outputs_ # Number of outputs when fit performed


model.predict(featues)




##Predicting on training data
pred_train = model.predict(x_train)
##Accuracy on training data
from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(y_train,pred_train)#1.0
##100%

##Confusion matrix
from sklearn.metrics import confusion_matrix
con_train = confusion_matrix(y_train,pred_train)

##Prediction on test data
pred_test = model.predict(x_test)

##Accuracy on test data
accuracy_test = accuracy_score(y_test,pred_test)#0.8
#80.0%
np.mean(y_test==pred_test)
# 0.8  ~   80%
##Confusion matrix
con_test = confusion_matrix(y_test,pred_test)



###### VISUALIZING THE ONE DECISION TREE IN RANDOM FOREST ####

from sklearn.tree import export_graphviz 
from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image  

predictors = colnames[0:10]
target = colnames[10]
tree1 = model.estimators_[20]
dot_data = StringIO()
export_graphviz(tree1,out_file = dot_data, feature_names =predictors, class_names = target, filled =True,rounded=True,impurity =False,proportion=False,precision =2)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

##Creating pdf file
graph.write_pdf('companyrf.pdf')

##Creating png file
graph.write_png('companyrf.png')
Image(graph.create_png())


