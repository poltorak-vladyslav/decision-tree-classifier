import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier

from utilities import visualize_classifier

# Завантаження вхідних даних
input_file = 'data_decision_trees.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розділення вхідних даних на два класи за мітками
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])

# Візуалізація вхідних даних
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black',
        edgecolors='black', linewidth=1, marker='x')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
        edgecolors='black', linewidth=1, marker='o')
plt.title('Вхідні дані')

# Розділення даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.25, random_state=5)

# Класифікатор на основі дерев рішень
params = {'random_state': 0, 'max_depth': 4}
classifier = DecisionTreeClassifier(**params)
classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train, 'Навчальна вибірка')

y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, 'Тестова вибірка')

# Оцінка продуктивності класифікатора
class_names = ['Клас-0', 'Клас-1']
print("\n" + "#"*40)
print("\nПродуктивність класифікатора на навчальній вибірці\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

print("#"*40)
print("\nПродуктивність класифікатора на тестовій вибірці\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")

plt.show()