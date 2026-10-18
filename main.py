
import argparse
import csv
import math
import random
from collections import Counter
from pathlib import Path

FEATURES = ("budget", "camera")
LABELS = {"iPhone", "Android"}


def load_data(path):
    """Чтение CSV-файла и проверка входных данных."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = set(FEATURES) | {"phone"}
        if (reader.fieldnames or []) != [*FEATURES, "phone"]:
            raise ValueError(f"CSV должен содержать столбцы: {', '.join((*FEATURES, 'phone'))}")

        for line_number, item in enumerate(reader, start=2):
            try:
                features = tuple(float(item[column]) for column in FEATURES)
                raw_label = item["phone"].strip().lower()
                label = {"iphone": "iPhone", "android": "Android"}[raw_label]
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Некорректные данные в строке {line_number}") from error
            budget, camera = features
            if budget <= 0 or not (1 <= camera <= 5):
                raise ValueError(f"Значение вне допустимого диапазона в строке {line_number}")
            rows.append((features, label))

    count = Counter(label for _, label in rows)
    if any(count[label] < 5 for label in LABELS):
        raise ValueError("Необходимо минимум 5 человек для каждого типа телефона; рекомендуется 20 и более.")
    return rows


def stratified_split(rows, seed=42):
    """Разделение данных на 60% для обучения, 20% для валидации и 20% для тестирования в каждом классе."""
    rng = random.Random(seed)
    train, validation, test = [], [], []
    for label in sorted(LABELS):
        group = [row for row in rows if row[1] == label]
        rng.shuffle(group)
        n = len(group)
        n_val = max(1, round(0.2 * n))
        n_test = max(1, round(0.2 * n))
        n_train = n - n_val - n_test
        train.extend(group[:n_train])
        validation.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def fit_scaler(train):
    """Вычисление параметров нормализации только по обучающей выборке (во избежание утечки данных)."""
    minimums = [min(row[0][j] for row in train) for j in range(len(FEATURES))]
    maximums = [max(row[0][j] for row in train) for j in range(len(FEATURES))]
    ranges = [hi - lo if hi != lo else 1 for lo, hi in zip(minimums, maximums)]
    return minimums, ranges


def scale(point, minimums, ranges):
    return tuple((x - lo) / span for x, lo, span in zip(point, minimums, ranges))


def normalize_rows(rows, minimums, ranges):
    return [(scale(point, minimums, ranges), label) for point, label in rows]


def knn_predict(train, point, k):
    """Прогнозирование методом k ближайших соседей с голосованием большинства."""
    if not 1 <= k <= len(train):
        raise ValueError("k должно быть от 1 до количества элементов обучающей выборки")
    ordered = sorted(train, key=lambda row: math.dist(row[0], point))
    neighbors = ordered[:k]
    votes = Counter(label for _, label in neighbors)
    highest = max(votes.values())
    # При равенстве голосов выбирается класс ближайшего соседа среди классов с одинаковым числом голосов.
    return next(label for _, label in neighbors if votes[label] == highest)


def accuracy(train, items, k):
    correct = sum(knn_predict(train, point, k) == label for point, label in items)
    return correct / len(items)


def main():
    parser = argparse.ArgumentParser(description="Классификация iPhone/Android с помощью метода KNN")
    parser.add_argument("--data", default=str(Path(__file__).with_name("data.csv")),
                        help="Путь к CSV-файлу")
    parser.add_argument("--budget", type=float, default=20, help="Бюджет (млн вьетнамских донгов)")
    parser.add_argument("--camera", type=float, default=4, help="Важность камеры (1-5)")
    args = parser.parse_args()
    if args.budget <= 0 or not (1 <= args.camera <= 5):
        parser.error("budget должен быть > 0; camera должна находиться в диапазоне 1-5")

    raw_train, raw_val, raw_test = stratified_split(load_data(Path(args.data)))
    minimums, ranges = fit_scaler(raw_train)
    train = normalize_rows(raw_train, minimums, ranges)
    val = normalize_rows(raw_val, minimums, ranges)
    test = normalize_rows(raw_test, minimums, ranges)
    print(f"Количество образцов: обучение={len(train)}, валидация={len(val)}, тестирование={len(test)}")

    candidates = [k for k in (1, 3, 5, 7, 9) if k <= len(train)]
    scores = {k: accuracy(train, val, k) for k in candidates}
    for k, value in scores.items():
        print(f"k={k}: точность на валидационной выборке = {value:.1%}")
    best_k = min(candidates, key=lambda k: (-scores[k], k))
    print(f"Выбрано k={best_k}, так как точность на валидационной выборке максимальна (при равенстве выбирается меньшее k).")
    print(f"Точность на ТЕСТОВОЙ выборке (оценивается только один раз) = {accuracy(train, test, best_k):.1%}")

    new_point = scale((args.budget, args.camera), minimums, ranges)
    print("Прогноз для нового пользователя:", knn_predict(train, new_point, best_k))


if __name__ == "__main__":
    main()
