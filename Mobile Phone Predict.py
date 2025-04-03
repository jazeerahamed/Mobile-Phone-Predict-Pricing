pip install xgboost


#Basic Operations
import pandas as pd 
import numpy as np 

#visualizing libraries
import matplotlib.pyplot as plt 
import seaborn as sns 


#data preprocessing
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import RandomizedSearchCV 


#Model
from sklearn.neighbors import KNeighborsClassifier 
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 
from sklearn.ensemble import StackingClassifier


#evaluators
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score,classification_report


import warnings
warnings.filterwarnings('ignore') 

df= pd.read_csv('https://raw.githubusercontent.com/Datawithabdalah/Mobile-Price-Range-Prediction/main/data_mobile_price_range.csv')
df.head(
df.shape
df.size
df.isnull().sum()
df.info()
df.describe().T
df['price_range'].value_counts()
df['Pixels Dimension']=df['px_height']*df['px_width']
df.drop(columns=['px_height','px_width'],inplace=True)
df['Screen Dimension'] = df['sc_h']* df['sc_w']
df.drop(columns=['sc_h','sc_w'],inplace=True)
df.rename(columns={'battery_power':'Battery','blue':'Bluetooth','clock_speed':'Clock_Speed','dual_sim':'Dual_Sim','fc':'Front_Camera','four_g':'4G','int_memory':'Rom','m_dep':'Mobile_Depth','mobile_wt':'Mobile_weight',
                   'n_cores':'Number_of_cores','pc':'Primary_Camera','ram':'Ram','talk_time':'Talk_Time','three_g':'3G','touch_screen':'Touch_Screen','wifi':'Wi-Fi','price_range':'Price_range'},inplace=True)

df = df[['Battery', 'Bluetooth', 'Clock_Speed', 'Dual_Sim', 'Front_Camera', '4G',
       'Rom', 'Mobile_Depth', 'Mobile_weight', 'Number_of_cores',
       'Primary_Camera', 'Ram', 'Talk_Time', '3G', 'Touch_Screen', 'Wi-Fi',
        'Pixels Dimension', 'Screen Dimension','Price_range']]
plt.title('Distribution of 4G phones')
df['4G'].value_counts().plot(kind='pie',autopct='%.2f',labels=['4G Support','Not Support'],startangle=120)
plt.show()

plt.title('Distribution of 3G phones')
colors =['#4F6272', '#B7C3F3']
df['3G'].value_counts().plot(kind='pie',autopct='%.2f',labels=['3G Support','Not Support'],startangle=120,colors=colors)
plt.show()

plt.title('Distribution of Wi-Fi phones')
colors = ['#DD7596', '#8EB897']
df['Wi-Fi'].value_counts().plot(kind='pie',autopct='%.2f',labels=['Wi-Fi Support','Not Support'],startangle=120,colors = colors)
plt.show()

sns.boxplot(x='Price_range',y='Battery',data=df)
plt.title('Battery power vs Price Range')
plt.show()

plt.figure(figsize=(10,6))
df['Front_Camera'].hist(alpha=0.5,color='blue',label='Front camera')
df['Primary_Camera'].hist(alpha=0.5,color='red',label='Primary camera')
plt.title("No of Phones vs Camera megapixels of front and primary camera")
plt.legend()
plt.xlabel('MegaPixels')

sns.pairplot(df)
plt.figure(figsize=(15,10))
plt.title('Check Multicollinearity')
sns.heatmap(df.iloc[:,0:-1].corr(),annot=True)
plt.show()

dependent_variable ='Price_range'
independent_varaible = list(set(df.columns.tolist())-{dependent_variable})


x=df[independent_varaible].values
y=df[dependent_variable].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0,stratify=y)

stand = StandardScaler()
x_train = stand.fit_transform(x_train)
x_test = stand.transform(x_test)

error=[]
for i in range(1,1000):
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(x_train,y_train)
  knn_pred = knn.predict(x_test)
  error.append(np.mean(knn_pred!=y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,1000),error,color='black',linestyle='dashed',marker='o',markerfacecolor='red')
plt.title('Error rate vs  K')
plt.xlabel('k value')
plt.ylabel('Error')
plt.show()
print("minimum error ",min(error),"at the value of k =",error.index(min(error))+1)
knn = KNeighborsClassifier(n_neighbors=273,p=1,weights='distance',metric= 'manhattan')

knn.fit(x_train,y_train)

knn_pred = knn.predict(x_test)

knn_accuracy = accuracy_score(knn_pred,y_test)
knn_accuracy

nb=GaussianNB()
nb.fit(x_train,y_train)
nb_pred = nb.predict(x_test)

nb_accuracy = accuracy_score(nb_pred,y_test)
nb_accuracy

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
  "min_child_weight" : [ 1, 3, 5, 7 ],
  "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]

 }
xgb= XGBClassifier()


random_search= RandomizedSearchCV(xgb,param_distributions=params,n_iter=5,scoring='accuracy',n_jobs=-1,cv=5,verbose=3)
random_search.fit(x_train,y_train)
{'min_child_weight': 1,
 'max_depth': 5,
 'learning_rate': 0.25,
 'gamma': 0.1,
 'colsample_bytree': 0.7}

random_search.best_estimator_
xgb_pred = random_search.predict(x_test)
xg_boost_accuracy = accuracy_score(xgb_pred,y_test)
xg_boost_accuracy

svc_params = {'C':range(5,10,2),
              'kernel':['rbf','linear', 'poly', 'sigmoid',]}

svc_cv  = GridSearchCV(SVC(),param_grid=svc_params,cv=5,verbose=True,scoring='accuracy',n_jobs=-1)
svc_cv.fit(x_train,y_train)


