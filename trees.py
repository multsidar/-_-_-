import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

df = pd.read_csv('star_classification.csv')


plots=[]
for i in ['rerun_ID']:
    g=sns.relplot(data=df,x='obj_ID', y=i, hue='class')
    plt.show()

enc = OrdinalEncoder()
df['class'] = enc.fit_transform(df[['class']])
df['class'].head(10)


X = df.drop(columns=['class'])
y = df.loc[:, ['class']]
minmax = MinMaxScaler()
scaled = minmax.fit_transform(X)


best_feature = SelectKBest(score_func=chi2)
fit = best_feature.fit(scaled, y)


feature_score = pd.DataFrame({
    'feature' : X.columns,
    'score': fit.scores_
})


feature_score.sort_values(by=['score'], ascending=False, inplace=True)




std = StandardScaler()
scaled = std.fit_transform(X)
scaled = pd.DataFrame(scaled, columns=X.columns)



data_standardization = y.join(scaled)


X = data_standardization.loc[:, ['redshift','u', 'g', 'r', 'i', 'z']]
y = data_standardization.loc[:, 'class']


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)

print(dt.score(x_train,y_train),dt.score(x_test, y_test))
print(f'train: {round(dt.score(x_train,y_train) * 100, 2)}%')
print(f'test: {round(dt.score(x_test, y_test) * 100, 2)}%')


dt = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=20, min_samples_leaf=5)
dt.fit(x_train, y_train)

print(dt.score(x_train, y_train), dt.score(x_test, y_test))
print(f'train: {round(dt.score(x_train, y_train) * 100, 2)}%')
print(f'test: {round(dt.score(x_test, y_test) * 100, 2)}%')

plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=2, filled=True, feature_names=['redshift','u', 'g', 'r', 'i', 'z'])
plt.show()

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, random_state=42)

scores = cross_validate(rf, x_train, y_train, cv=2, return_train_score=True, n_jobs=-1, verbose=2)


print(np.mean(scores['train_score']), np.mean(scores['test_score']))
print(f'train: {round(np.mean(scores[r"train_score"]) * 100, 2)}%')
print(f'test: {round(np.mean(scores[r"test_score"]) * 100, 2)}%')


dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(x_train,y_train)

print(dt.score(x_train,y_train),dt.score(x_test, y_test))
print(f'train: {round(dt.score(x_train,y_train) * 100, 2)}%')
print(f'test: {round(dt.score(x_test, y_test) * 100, 2)}%')

plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['redshift','u', 'g', 'r', 'i', 'z'])
plt.show()
