import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sns
import numpy as np
import xlwings as xw



bike=pd.read_excel('train.csv')
bike.info()
bike.isna().sum()
bike['datetime']=pd.to_datetime(bike.datetime)
bike.season.value_counts()
bike['MonthofYear']=bike.datetime.dt.month
bike.drop(columns=['registered','casual'],inplace=True)


# Use excelwings to manipulate the excel workbook
book=xw.Book('Report_dashboard.xlsxm')
Sheet=book.sheets('Exploratory Analysis')

# Creating the plots
fig,(ax0,ax1,ax2)=plt.subplots(nrows=1,ncols=3)
fig.set_figheight(6)
fig.set_figwidth(21)
sns.boxplot(x='MonthofYear',y='count',data=bike,ax=ax0)
sns.regplot(x='atemp',y='count',scatter=True,scatter_kws={'alpha':0.05},x_jitter=True,y_jitter=True,data=bike,ax=ax1)
sns.histplot(x='count',data=bike,ax=ax2)
ax0.set_title('Bike Rentals by month')
ax0.set_ylabel('Bike Rentals')
ax1.set_title('Temp vs Bike rentals')
ax1.set_ylabel('Bike rentals')
ax2.set_label('Bike Rentals')
ax2.set_title('Distribution of Bike Rentals')
plt.tight_layout()
sns.set_context('talk')
#sns.set_theme('dark')
plt.show()
Sheet.pictures.add(fig,name='Bike Rentals',update=True)

corr_mat=bike.corr().round(2)
corr_mat
fig,ax=plt.subplots()
fig.set_figheight(7)
fig.set_figwidth(14)
sns.heatmap(corr_mat,annot=True,linewidth=0.3,cmap='viridis')
ax.set_title('Correlation Matrix')
Sheet.pictures.add(fig,name='Corr_mat',update=True,left=Sheet.range('B37').left,top=Sheet.range('B37').top)


X=bike[['atemp','MonthofYear','humidity','weather','holiday']]
y=bike['count']
bike.holiday.value_counts()
y


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
max_depths=[3,4,5,6,7,8,9,10,11,12,13,14,15]
for max_depth in max_depths:
    dt=DecisionTreeRegressor(max_depth=max_depth,random_state=20)
    dt.fit(X_train,y_train)
    y_pred=dt.predict(X_test)
    score=r2_score(y_test,y_pred)
    print('The r2_score of tree with max depth {} is'.format(max_depth),score)
    
dt=DecisionTreeRegressor(max_depth=4)
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)
sheet2=book.sheets('Decision tree')

import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

# Assuming you have an Index object named 'feature_names'
feature_names = pd.Index(['atemp', 'MonthofYear', 'humidity', 'weather', 'holiday'])

# Convert the Index object to a list
feature_names_list = feature_names.tolist()

# Create and fit the decision tree model
model = tree.DecisionTreeClassifier()
model.fit(X, y)


from sklearn import tree
fig,ax=plt.subplots()
fig.set_figheight(35)
fig.set_figwidth(35)
_=tree.plot_tree(dt,feature_names=feature_names_list,filled=True)

sheet2.pictures.add(fig,name='Tree',update=True,left=sheet2.range('A6').left,top=sheet2.range('A6').left)

type(fig)

dt=DecisionTreeRegressor(max_depth=7)
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)
sheet2.range('B2').value=dt.max_depth
sheet2.range('B3').value=r2_score(y_test,y_pred).round(2)

#Now trying Random forest
#We will probably not know the number of registered users ahead of time so drop the columns casual and registered to make this problem interestins
#We can use random forest to get feature importances and use it as a dimensionality reduction technique

#bike.drop(columns=['registered','casual'],inplace=True)
X=bike.drop(columns=['count','datetime'])
X.info()
y=bike['count']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train

from sklearn.ensemble import RandomForestRegressor
for max_depth in max_depths:
    rf=RandomForestRegressor(n_estimators=100,max_depth=max_depth,random_state=30)
    rf.fit(X_train,y_train)
    y_pred=rf.predict(X_test)
    score=r2_score(y_test,y_pred)
    print('R2_score with {} max depth is'.format(max_depth),score)
    
rf=RandomForestRegressor(n_estimators=100,max_depth=11,random_state=30)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
score=r2_score(y_test,y_pred)
sheet3=book.sheets('Random forest')
sheet3.range('B4').value=score.round(2)
sheet3.range('B2').value=rf.n_estimators
sheet3.range('B3').value=rf.max_depth

mat=pd.DataFrame(rf.feature_importances_,index=X.columns).sort_values(by=[0])
fig,ax=plt.subplots()
fig.set_figheight(7)
fig.set_figwidth(12)
ax.barh(width=mat[0],y=mat.index)
plt.xticks(rotation=45)
ax.set_title('Feature Importances')
sheet3.pictures.add(fig,name='Feature',update=True,left=sheet3.range('A8').left,top=sheet3.range('A8').left)

type(mat)
'''
'''   