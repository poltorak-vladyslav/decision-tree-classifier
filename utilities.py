import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y, title=''):
    # Визначення мінімальних та максимальних значень для X та Y,
    # які будуть використані у створенні сітки
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Визначення розміру кроку для побудови сітки
    mesh_step_size = 0.01

    # Створення сітки значень X та Y
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))

    # Запуск класифікатора на сітці
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    # Зміна форми вихідного масиву
    output = output.reshape(x_vals.shape)

    # Створення графіка
    plt.figure()

    # Задання заголовку
    plt.title(title)

    # Вибір колірної схеми для графіка
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)

    # Накладання навчальних точок на графік
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # Вказання меж графіка
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())

    # Вказання поділок на осях X та Y
    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))

    plt.show()