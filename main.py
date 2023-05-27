import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

df = pd.read_csv('star_classification.csv')

print(df.head())


print(df.info())


'''
plots=[]
for i in ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'run_ID',
        'cam_col', 'field_ID', 'spec_obj_ID', 'redshift',
       'plate', 'MJD', 'fiber_ID']:
    g=sns.relplot(data=df,x='obj_ID', y=i, hue='class')
    plots.append(g);
'''
enc = OrdinalEncoder()
df['class'] = enc.fit_transform(df[['class']])
df['class'].head(10)


X = df.drop(columns=['class'])
y = df.loc[:, ['class']]
minmax_scale = MinMaxScaler()
scaled = minmax_scale.fit_transform(X)


feature = SelectKBest(score_func=chi2)
feafit = feature.fit(scaled, y)


feature_score = pd.DataFrame({
    'feature' : X.columns,
    'score': feafit.scores_
})


feature_score.sort_values(by=['score'], ascending=False, inplace=True)
print(feature_score)



std = StandardScaler()
scaled = std.fit_transform(X)
scaled = pd.DataFrame(scaled, columns=X.columns)
print(scaled.head())


data_stand = y.join(scaled)


X = data_stand.loc[:, ['redshift', 'u', 'g', 'r', 'i', 'z']]
y = data_stand.loc[:, 'class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


k_neighbors = 30
metrics = ['euclidean', 'manhattan']


accuracys = []
for k in range(1, k_neighbors+1, 1):
    accuracy_k = []
    for metric in metrics:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric=metric)
        knn.fit(X_train, y_train)
        y_predict = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_predict)
        accuracy_k.append(accuracy)
    accuracys.append(accuracy_k)


accuracy_data = pd.DataFrame(np.array(accuracys), columns=metrics)
k_df = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], columns=['k'])
accuracy_1= k_df.join(accuracy_data)


plt.plot(accuracy_1['k'], accuracy_1['euclidean'], 'o', label='euclidean')
plt.plot(accuracy_1['k'], accuracy_1['manhattan'], 'o', label='manhattan')

plt.legend()
plt.xlabel('k')
plt.ylabel('accuracy score')
plt.show()


knn = KNeighborsClassifier(n_neighbors=13, weights='distance', metric='manhattan')
print(knn.fit(X_train, y_train))


y_predict = knn.predict(X_test)
print(accuracy_score(y_test, y_predict))

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

print(dt.score(X_train, y_train), dt.score(X_test, y_test))
print(f'train: {round(dt.score(X_train, y_train) * 100, 2)}%')
print(f'test: {round(dt.score(X_test, y_test) * 100, 2)}%')


dt = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=20, min_samples_leaf=5)
dt.fit(X_train, y_train)

print(dt.score(X_train, y_train), dt.score(X_test, y_test))
print(f'train: {round(dt.score(X_train, y_train) * 100, 2)}%')
print(f'test: {round(dt.score(X_test, y_test) * 100, 2)}%')
accuracy_tr = []
par = []
for i in range(1,9):
    dt = DecisionTreeClassifier(random_state=i*10, max_depth=5, min_samples_split=20, min_samples_leaf=5)
    dt.fit(X_train, y_train)
    print('random_state='+ str(i*10) + ' max_depth=5, min_samples_split=20, min_samples_leaf=5')
    print(dt.score(X_train, y_train), dt.score(X_test, y_test))
    print(f'train: {round(dt.score(X_train, y_train) * 100, 2)}%')
    print(f'test: {round(dt.score(X_test, y_test) * 100, 2)}%')
    accuracy_tr.append(dt.score(X_test, y_test))
    par.append(i*10)
plt.plot(par,accuracy_tr,'o', label='random_state')

accuracy_tr2 = []
par2 = []
for i in range(1,9):
    dt = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=i*10, min_samples_leaf=5)
    dt.fit(X_train, y_train)
    print('min_samples_split'+ str(i*10) + ' random_state=42, max_depth=5, min_samples_leaf=5')
    print(dt.score(X_train, y_train), dt.score(X_test, y_test))
    print(f'train: {round(dt.score(X_train, y_train) * 100, 2)}%')
    print(f'test: {round(dt.score(X_test, y_test) * 100, 2)}%')
    accuracy_tr2.append(dt.score(X_test, y_test))
    par2.append(i*10)
plt.plot(par2,accuracy_tr2,'o',label='min_samples_split')

accuracy_tr3 = []
par3 = []
for i in range(1,9):
    dt = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=20, min_samples_leaf=i*10)
    dt.fit(X_train, y_train)
    print('min_samples_split'+ str(i) + ' random_state=42, max_depth=5, min_samples_split=20')
    print(dt.score(X_train, y_train), dt.score(X_test, y_test))
    print(f'train: {round(dt.score(X_train, y_train) * 100, 2)}%')
    print(f'test: {round(dt.score(X_test, y_test) * 100, 2)}%')
    accuracy_tr3.append(dt.score(X_test, y_test))
    par3.append(i*10)
plt.plot(par3,accuracy_tr3,'o',label='min_samples_leaf')
plt.legend()

plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=2, filled=True, feature_names=['redshift','u', 'g', 'r', 'i', 'z'])
plt.show()

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, random_state=42)

scores = cross_validate(rf, X_train, y_train, cv=2, return_train_score=True, n_jobs=-1, verbose=2)


print(np.mean(scores['train_score']), np.mean(scores['test_score']))
print(f'train: {round(np.mean(scores[r"train_score"]) * 100, 2)}%')
print(f'test: {round(np.mean(scores[r"test_score"]) * 100, 2)}%')


dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

print(dt.score(X_train, y_train), dt.score(X_test, y_test))
print(f'train: {round(dt.score(X_train, y_train) * 100, 2)}%')
print(f'test: {round(dt.score(X_test, y_test) * 100, 2)}%')

plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['redshift','u', 'g', 'r', 'i', 'z'])
plt.show()

