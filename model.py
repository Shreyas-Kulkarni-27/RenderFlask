import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.svm import SVR
import seaborn as sns
import pickle

df=pd.read_csv('HR-Employee-Attrition.csv')

df.head()

df.describe()

df.info()

df['Attrition'].replace('Yes','1',inplace=True)
df['Attrition'].replace('No','0',inplace=True)

df.iloc[:,0:10]
df.iloc[:,20:30]
df.iloc[:,10:20]

df['StandardHours'].nunique()
df.iloc[:,30:36]

df.columns

df.info()

df.drop(['EmployeeCount','Over18','StandardHours','EmployeeNumber'],axis=1,inplace=True)

cat=[]
num=[]
for i in df.columns:
    if df[i].dtype == "object":
        cat.append(i)
    else:
        num.append(i)

print(cat)
print(num)

df.duplicated().sum()

for i in cat:
    if df[i].dtype == "object":
        print(i.upper(),': ',df[i].nunique())
        print(df[i].value_counts().sort_values())
        print('\n')

df.shape

plt.hist(df['Age'], bins=10)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Age')
plt.show()

df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 30, 40, 50, 60], labels=['18-30','31-40','41-50','51-60'])
age_group_counts = df['AgeGroup'].value_counts()
print(age_group_counts)

age_group_attrition_counts = df[df['Attrition'] == '1'].groupby('AgeGroup')['Attrition'].count()
age_group_attrition_counts

age_group_counts = df['AgeGroup'].value_counts()
age_group_attrition_counts = df[df['Attrition'] == '1'].groupby('AgeGroup')['Attrition'].count()
plt.figure(figsize=(10, 6))

age_group_counts = age_group_counts.reindex(age_group_attrition_counts.index)

age_group_counts.plot(kind='bar', color='skyblue', label='Total Count')
age_group_attrition_counts.plot(kind='bar', color='orange', label='Attrition Count')

for i,(total, attrition) in enumerate(zip(age_group_counts, age_group_attrition_counts)):
    percentage = attrition / total * 100
    plt.text(i, total + 10, f'{percentage:.2f}%', ha = 'center')

cat_df=df.select_dtypes(include='object')

for i in cat_df:
    plt.figure(figsize=(15, 15))
    sns.catplot(data=df,x=i,kind='count')

k=1
plt.figure(figsize=(40, 40))
for col in df:
  if col=="Attrition":
    continue
  yes = df[df['Attrition'] == 'Yes'][col]
  no = df[df['Attrition'] == 'No'][col]
  plt.subplot(6, 6, k)
  plt.hist(yes, bins=25, alpha=0.5, label='yes', color='b')
  plt.hist(no, bins=25, alpha=0.5, label='no', color='r')
  plt.legend(loc='upper right')
  plt.title(col)
  k+=1

  business_travel_counts = df['BusinessTravel'].value_counts()
business_travel_counts

categories =  ['Travel_Rarely' , 'Travel_Frequently', 'Non-Travel']

colors = {'Travel_Rarely': ['#FF7F50','#32CD32'],
          'Travel_Frequently': ['#FF7F50','#32CD32'],
          'Non-Travel': ['#FF7F50','#32CD32']}

fig, axes = plt.subplots(1,3, figsize=(15,5))

for i, category in enumerate(categories):
    category_data = df[df['BusinessTravel'] == category]

    attrition_counts = category_data['Attrition'].value_counts()
    total_counts = category_data['Attrition'].count()
    percentage_yes = (attrition_counts['1'] / total_counts)*100

    wedges, texts, autotexts = axes[i].pie([percentage_yes, 100 - percentage_yes], colors=colors[category],autopct='%1.1f%%',startangle=90)
    axes[i].set_title(f'{category}Attrition')
    axes[i].axis('equal')

    legend_labels = ['Yes','No']
    axes[i].legend(wedges,legend_labels,loc='center',bbox_to_anchor=(0.5,-0.1), fancybox=True, shadow=True,ncol=2)

plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x='Attrition',y='TotalWorkingYears',data=df)
plt.title('Bivariate Analysis: Attrition vs. Total Working Years')
plt.xlabel('Attrition')
plt.ylabel('Total Working Years')
plt.show()

plt.figure(figsize=(15,10))
sns.heatmap(df[num].corr(),annot=True)
plt.show()

plt.pie(df['Attrition'].value_counts(),labels=['No','Yes'],autopct='%.0f%%')
plt.show()

df.drop('AgeGroup',axis=1,inplace=True)

table=pd.crosstab(df.JobSatisfaction, df.Attrition)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Job satisfaction vs attrition')

table=pd.crosstab(df.OverTime, df.Attrition)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Overtime vs attrition')

table=pd.crosstab(df.BusinessTravel, df.Attrition)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Business Travel vs attrition')

table=pd.crosstab(df.YearsSinceLastPromotion, df.Attrition)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Business Travel vs attrition')

a4_dims = (25, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.countplot(data=df,x="JobRole",hue="Attrition", ax=ax )

plt.pie(df['Attrition'].value_counts(),labels=['No','Yes'],autopct='%.0f%%')
plt.show()

from sklearn.preprocessing import LabelEncoder
for column in df.columns:
        if df[column].dtype == np.number:
            continue
        else:
            df[column] = LabelEncoder().fit_transform(df[column])

X = df.drop('Attrition', axis=1)

Y = df['Attrition']

for feature in X.columns:
    if X[feature].dtype == 'object':
        print('\n')
        print('feature: ',feature)
        print(pd.Categorical(X[feature].unique()))
        print(pd.Categorical(X[feature].unique()).codes)
        X[feature] = pd.Categorical(X[feature]).codes

X.head()

from scipy.stats import zscore

X = X.apply(zscore)
X.head()
Y.head(2)
Y = Y.astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30, random_state=1, stratify=Y)

from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()
logReg.fit(X_train, Y_train)
y_pred = logReg.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:")
print(cm)

from sklearn.tree import DecisionTreeClassifier
DecTree = DecisionTreeClassifier()
DecTree.fit(X_train, Y_train)
y_pred = DecTree.predict(X_test)

accuracy1 = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth' : [5,6,7],
    'min_samples_leaf' : [7,10,20,30,40],
    'min_samples_split' : [30,40,60,90,120],
}
dtHyper = DecisionTreeClassifier(random_state=1)
grid_search = GridSearchCV(estimator= dtHyper, param_grid=param_grid, cv=10)
grid_search.fit(X_train,Y_train)
grid_search.best_params_
best_grid = grid_search.best_estimator_
ytrain_dtpredict = best_grid.predict(X_train)
print(ytrain_dtpredict)
ytest_dtpredict = best_grid.predict(X_test)
print(ytest_dtpredict)

cf_matrix=confusion_matrix(Y_train, ytrain_dtpredict)
sns.heatmap(cf_matrix, square=True,annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

logreg1 = LogisticRegression()
param_grid = {
    'penalty' : ['l2'],
    'C' : [0.1,1.0,10.0]
}
grid_search1 = GridSearchCV(estimator=logreg1,param_grid=param_grid,cv=5, error_score='raise')
grid_search1.fit(X_train, Y_train)
best_model = grid_search.best_estimator_
print("Model prediction done")
print(best_model )

ytrain_dt1predict = best_model.predict(X_train)
unique_counts, count_ones = np.unique(ytrain_dt1predict,return_counts=True)
print(dict(zip(unique_counts,count_ones)))
print(ytrain_dt1predict)

ytest_dt1predict = best_model.predict(X_test)
unique_counts, count_ones = np.unique(ytest_dt1predict,return_counts=True)
print(dict(zip(unique_counts,count_ones)))
print(ytest_dt1predict)

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=100, random_state=42)
RF.fit(X_train, Y_train)
y_pred = RF.predict(X_train)
print(y_pred)
count_ones = np.count_nonzero(y_pred == 1)
print(count_ones)
total_elements = len(y_pred)
print(total_elements)

feature_importances_ = RF.feature_importances_
feature_imp = pd.Series(feature_importances_, index=list(X.columns)).sort_values(ascending=False)
print(feature_imp)

# sns.set_theme(rc={'figure.figsize':(13,9)})
sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Important Features")
plt.savefig('important_features.png')
plt.show()

pickle.dump(logReg, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))