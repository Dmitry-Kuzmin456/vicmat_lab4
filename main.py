import numpy as np
import matplotlib.pyplot as plt
from models import ApproximationModels, get_rms

# Словарь вариантов заданий
VARIANTS = {
    "1": {"name": "3x / (x^4 + 3)", "func": lambda x: (3 * x) / (x ** 4 + 3), "range": (-2, 0)},
    "2": {"name": "sin(x) + 0.5", "func": lambda x: np.sin(x) + 0.5, "range": (0, 3)},
    "3": {"name": "x^2 - 4x", "func": lambda x: x ** 2 - 4 * x, "range": (0, 4)}
}


def main():
    print("--- Аппроксимация функций МНК ---")
    print("1. Загрузить данные из data.txt")
    print("2. Выбрать функцию из вариантов")

    choice = input("Ваш выбор: ")

    if choice == "2":
        print("\nДоступные варианты:")
        for k, v in VARIANTS.items():
            print(f"{k}. {v['name']} на интервале {v['range']}")
        v_idx = input("Номер варианта: ")
        var = VARIANTS.get(v_idx, VARIANTS["1"])

        x = np.linspace(var['range'][0], var['range'][1], 11)
        y = var['func'](x)
        print(f"Сгенерированы данные для функции {var['name']}")
    else:
        try:
            data = np.loadtxt('data.txt')
            x, y = data[:, 0], data[:, 1]
        except Exception as e:
            print(f"Ошибка загрузки файла: {e}")
            return

    models = [
        ("Линейная", ApproximationModels.linear),
        ("Полином 2 ст.", ApproximationModels.poly_2),
        ("Полином 3 ст.", ApproximationModels.poly_3),
        ("Экспоненциальная", ApproximationModels.exponential),
        ("Логарифмическая", ApproximationModels.log),
        ("Степенная", ApproximationModels.power)
    ]

    results = []
    plt.figure(figsize=(12, 7))
    plt.scatter(x, y, color='black', zorder=5, label='Исходные данные')

    # Для плавных графиков
    x_plot = np.linspace(min(x) - 0.2, max(x) + 0.2, 300)

    print(f"\n{'Модель':<20} | {'RMS (СКО)':<10} | {'S (Мера)':<10} | {'Формула'}")
    print("-" * 80)

    for name, factory in models:
        res = factory(x, y)
        if res is None: continue

        # Распаковка (у линейной есть Пирсон, у остальных нет)
        pearson = None
        if len(res) == 3:
            func, formula, pearson = res
        else:
            func, formula = res

        y_pred = func(x)
        rms, s_val = get_rms(y, y_pred)
        results.append({"name": name, "formula": formula, "rms": rms, "s": s_val})

        p_str = f" [r={pearson:.3f}]" if pearson is not None else ""
        print(f"{name:<20} | {rms:<10.4f} | {s_val:<10.4f} | {formula}{p_str}")

        plt.plot(x_plot, func(x_plot), label=f"{name} (RMS:{rms:.3f})", alpha=0.8)

    best_model = min(results, key=lambda t: t['rms'])
    print("-" * 80)
    print(f"НАИЛУЧШАЯ МОДЕЛЬ: {best_model['name']}")
    print(f"ФОРМУЛА: {best_model['formula']}")
    print(f"МИНИМАЛЬНОЕ СКО: {best_model['rms']:.4f}")

    plt.title("Аппроксимация функций методом наименьших квадратов")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    main()