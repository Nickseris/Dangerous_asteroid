import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree


df = pd.read_csv("nasa.csv")

# удаление ненужных столбцов
del df['Close Approach Date']
del df['Orbiting Body']
del df['Orbit Determination Date']
del df['Equinox']

def fill_danger(row):
    if row['Hazardous'] == True:
        return 1
    else:
        return 0


'''Заполнение целевой переменной числами, а не булевыми значениями'''
df['Hazardous'] = df.apply(fill_danger, axis = 1)

# отделим целевую переменную от остальгных даанных
x = df.drop("Hazardous", axis = 1)
y = df['Hazardous']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

model = tree.DecisionTreeClassifier(criterion="entropy")
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
percents = accuracy_score(y_test, y_pred) * 100
print(percents)