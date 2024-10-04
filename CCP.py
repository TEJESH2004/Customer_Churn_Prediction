import pandas as pd
data = pd.read_csv('Churn_Modelling.csv')
data.head()
data.shape
data.info()
data.isnull()
data.isnull().sum()
data.describe()
data.columns
data=data.drop(['RowNumber','CustomerId','Surname'],axs=1)
data.head()
data['Geography'].unique()
data = pd.get_dummies(data,drop_first=True)
data.head()
data['Exited'].value_counts()
import seaborn as sns
sns.countplot(data['Exited'])
# # Handeling imbalanced data with SMOT
from imblearn.over_sampling import SMOTE
x_res,y_res =SMOTE().fit_resample(x,y)
y_res.value_counts()
x = data.drop('Exited',axis=1)
y = data['Exited']
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42,stratify=y)
# # feature scalling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train=sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_train
# # logistic regression
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x_train,y_train)
y_pred1 =log.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred1)
from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(y_test,y_pred1)
recall_score(y_test,y_pred1)
f1_score(y_test,y_pred1)
# # SVC
from sklearn import svm
svm = svm.SVC()
# f1_score(y_test,y_pred1)
svm.fit(x_train,y_train)
y_pred2 = svm.predict(x_test)
accuracy_score(y_test,y_pred2)
precision_score(y_test,y_pred2)
f1_score(y_test,y_pred2)
# # KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred3 = knn.predict(x_test)
accuracy_score(y_test,y_pred3)
precision_score(y_test,y_pred3)
f1_score(y_test,y_pred3)
# # Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred4 = dt.predict(x_test)
accuracy_score(y_test,y_pred4)
precision_score(y_test,y_pred4)
f1_score(y_test,y_pred4)
# # RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
rf =RandomForestClassifier()
rf.fit (x_train,y_train)
y_pred5 = rf.predict(x_test)
accuracy_score(y_test,y_pred5)
precision_score(y_test,y_pred5)
f1_score(y_test,y_pred5)
# # Gradient boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(x_train,y_train)
y_pred6 = gbc.predict(x_test)
accuracy_score(y_test,y_pred6)
precision_score(y_test,y_pred6)
f1_score(y_test,y_pred6)
final_data=pd.DataFrame({ 'Models':['LR','SVC','KNN','DT','RF','GBC'],
                         'ACC':[accuracy_score(y_test,y_pred1),
                        accuracy_score(y_test,y_pred2),
                        accuracy_score(y_test,y_pred3),
                        accuracy_score(y_test,y_pred4),
                        accuracy_score(y_test,y_pred5),
                        accuracy_score(y_test,y_pred6)] })
final_data
import seaborn as sns
sns.barplot(final_data['Models'],final_data['ACC'])
final_data=pd.DataFrame({ 'Models':['LR','SVC','KNN','DT','RF','GBC'],
                         'PRE':[precision_score(y_test,y_pred1),
                        precision_score(y_test,y_pred2),
                        precision_score(y_test,y_pred3),
                        precision_score(y_test,y_pred4),
                        precision_score(y_test,y_pred5),
                        precision_score(y_test,y_pred6)] })
final_data
sns.barplot(final_data['Models'],final_data['PRE'])

# # CREATING THE MODEL
x_rec=sc.fit_transform(x_res)
rf.fit(x_res,y_res)
import joblib
joblib.dump(rf,'churn_predict_model')
model = joblib.load('churn_predict_model')
model.predict([[619,42,2,0.0,0,0,0,101348.88,0,0,0]])
