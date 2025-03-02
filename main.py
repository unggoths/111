import math
import random

def use_distance_metric(metric_func):
    """
    Декоратор для застосування різних метрик відстані до функцій.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Зберігаємо оригінальну функцію, але додаємо метрику як атрибут
            wrapper.distance_metric = metric_func
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Реалізація різних метрик відстані
def euclidean_distance(x, y):
    """Обчислює евклідову відстань між двома векторами."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

def manhattan_distance(x, y):
    """Обчислює манкхеттенську відстань між двома векторами."""
    return sum(abs(a - b) for a, b in zip(x, y))

def cosine_distance(x, y):
    """Обчислює косинусну відстань між двома векторами."""
    dot = dot_product(x, y)
    norm_x = vector_norm(x)
    norm_y = vector_norm(y)
    if norm_x * norm_y == 0:  # Уникаємо ділення на нуль
        return 1.0  # Максимальна відстань, якщо одна з норм дорівнює 0
    return 1 - (dot / (norm_x * norm_y))

def chebyshev_distance(x, y):
    """Обчислює чебишевську відстань між двома векторами."""
    return max(abs(a - b) for a, b in zip(x, y))


def cosine_distance(point1, point2):
    """Косинусна відстань між двома точками (1 - косинусна подібність)."""
    dot_prod = dot_product(point1, point2)
    norm1 = vector_norm(point1)
    norm2 = vector_norm(point2)
    if norm1 * norm2 == 0:  # Уникаємо ділення на нуль
        return 1.0  # Максимальна відстань, якщо одна з норм дорівнює 0
    cosine_similarity = dot_prod / (norm1 * norm2)
    return 1 - cosine_similarity


@use_distance_metric(euclidean_distance)  # За замовчуванням використовуємо евклідову відстань
def knn(x_train, y_train, x_test, k):
    """
    Функція k-найближчих сусідів (k-Nearest Neighbors) для класифікації точок.

    Parameters:
    x_train -- список або список списків з координатами навчальних точок
    y_train -- список класів для навчальних точок
    x_test -- список або список списків з координатами тестових точок
    k -- кількість найближчих сусідів для голосування

    Returns:
    predictions -- список передбачених класів для кожної тестової точки
    """
    predictions = []
    distance_metric = knn.distance_metric  # Отримуємо метрику відстані з декоратора

    for test_point in x_test:
        # Обчислюємо відстані до всіх точок навчального набору
        distances = []
        for i, train_point in enumerate(x_train):
            dist = distance_metric(test_point, train_point)
            distances.append((dist, y_train[i]))  # Зберігаємо відстань і клас точки

        # Сортуємо відстані за зростанням
        distances.sort(key=lambda x: x[0])

        # Беремо k найближчих сусідів
        k_nearest = distances[:k]

        # Підраховуємо голоси для кожного класу серед k найближчих сусідів
        class_votes = {}
        for _, class_label in k_nearest:
            class_votes[class_label] = class_votes.get(class_label, 0) + 1

        # Знаходимо клас з найбільшою кількістю голосів
        predicted_class = max(class_votes.items(), key=lambda x: x[1])[0]
        predictions.append(predicted_class)

    return predictions


@use_distance_metric(euclidean_distance)  # За замовчуванням використовуємо евклідову відстань
def k_means(X, k, max_iterations):
    """
    Алгоритм k-середніх для кластеризації точок.

    Parameters:
    X -- список або список списків з координатами точок
    k -- кількість кластерів (центрів)
    max_iterations -- максимальна кількість ітерацій для конвергенції

    Returns:
    clusters -- список, що містить індекси кластерів для кожної точки
    centroids -- список координат центроїдів для кожного кластера
    """


    # 1. Ініціалізація: вибираємо k випадкових центроїдів
    n_features = len(X[0])  # Кількість ознак (наприклад, x, y)
    centroids = [X[random.randint(0, len(X) - 1)] for _ in range(k)]

    for iteration in range(max_iterations):
        # 2.1. Призначаємо кожну точку до найближчого центроїда
        clusters = []
        distance_metric = k_means.distance_metric  # Отримуємо метрику відстані з декоратора
        for point in X:
            min_dist = float('inf')
            cluster_idx = 0
            for i, centroid in enumerate(centroids):
                dist = distance_metric(point, centroid)
                if dist < min_dist:
                    min_dist = dist
                    cluster_idx = i
            clusters.append(cluster_idx)

        # 2.2. Оновлюємо центроїди як середнє значення всіх точок у кожному кластері
        new_centroids = []
        for i in range(k):
            cluster_points = [X[j] for j in range(len(X)) if clusters[j] == i]
            if cluster_points:  # Якщо є точки в кластері
                centroid = []
                for feature in range(n_features):
                    mean = sum(point[feature] for point in cluster_points) / len(cluster_points)
                    centroid.append(mean)
                new_centroids.append(centroid)
            else:
                new_centroids.append(centroids[i])

        # 3. Перевіряємо, чи змінилися центроїди (збіжність)
        centroids_changed = False
        for i in range(k):
            dist = distance_metric(new_centroids[i], centroids[i])
            if dist > 1e-6:  # Якщо зміна значна
                centroids_changed = True
                break

        centroids = new_centroids

        if not centroids_changed:
            break

    return clusters, centroids


# Приклад зміни метрики (коментар, щоб не виконувати автоматично)
"""
# Зміна метрики для k-NN на косинусну
knn = use_distance_metric(cosine_distance)(knn)

# Зміна метрики для k-means на манкхеттенську, наприклад
k_means = use_distance_metric(manhattan_distance)(k_means)
"""


def dot_product(vec1, vec2):
    """
    Обчислює скалярний добуток двох векторів.

    Parameters:
    vec1, vec2 -- списки чисел (вектори однакової довжини)

    Returns:
    Число -- скалярний добуток векторів
    """
    if len(vec1) != len(vec2):
        raise ValueError("Вектори повинні мати однакову довжину")
    result = 0
    for i in range(len(vec1)):
        result += vec1[i] * vec2[i]
    return result


def vector_subtract(vec1, vec2):
    """
    Віднімає другий вектор від першого.

    Parameters:
    vec1, vec2 -- списки чисел (вектори однакової довжини)

    Returns:
    Список -- результат віднімання (vec1 - vec2)
    """
    if len(vec1) != len(vec2):
        raise ValueError("Вектори повинні мати однакову довжину")
    return [vec1[i] - vec2[i] for i in range(len(vec1))]


def vector_add(vec1, vec2):
    """
    Додає два вектори.

    Parameters:
    vec1, vec2 -- списки чисел (вектори однакової довжини)

    Returns:
    Список -- результат додавання (vec1 + vec2)
    """
    if len(vec1) != len(vec2):
        raise ValueError("Вектори повинні мати однакову довжину")
    return [vec1[i] + vec2[i] for i in range(len(vec1))]


def vector_multiply(vec, scalar):
    """
    Множить вектор на скаляр.

    Parameters:
    vec -- список чисел (вектор)
    scalar -- число (скаляр)

    Returns:
    Список -- результат множення вектора на скаляр
    """
    return [x * scalar for x in vec]


def vector_norm(vec):
    """
    Обчислює евклідову норму вектора (довжину).

    Parameters:
    vec -- список чисел (вектор)

    Returns:
    Число -- норма вектора
    """
    squared_sum = sum(x * x for x in vec)
    return squared_sum ** 0.5


def vector_mean(vectors):
    """
    Обчислює середнє арифметичне набору векторів.

    Parameters:
    vectors -- список векторів (список списків чисел)

    Returns:
    Список -- вектор, що є середнім значенням вхідних векторів
    """
    if not vectors or not vectors[0]:
        raise ValueError("Список векторів не може бути порожнім")
    n_features = len(vectors[0])
    n_vectors = len(vectors)

    # Перевіряємо, чи всі вектори мають однакову довжину
    if not all(len(v) == n_features for v in vectors):
        raise ValueError("Усі вектори повинні мати однакову довжину")

    mean_vector = []
    for feature in range(n_features):
        mean = sum(vectors[i][feature] for i in range(n_vectors)) / n_vectors
        mean_vector.append(mean)
    return mean_vector