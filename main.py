import numpy as np
import matplotlib.pyplot as plt
from models import ApproximationModels, get_rms


def main():
    try:
        data = np.loadtxt('data.txt')
        x, y = data[:, 0], data[:, 1]
    except:
        print("Ошибка загрузки data.txt")
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
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='red', label='Исходные данные')

    x_plot = np.linspace(min(x) - 0.2, max(x) + 0.2, 200)

    for name, factory in models:
        res = factory(x, y)
        pearson = None
        if len(res) == 3:
            func, formula, pearson = res
        else:
            func, formula = res

        y_pred = func(x)
        rms, s_val = get_rms(y, y_pred)
        results.append((name, formula, rms, s_val, pearson))
        plt.plot(x_plot, func(x_plot), label=f"{name}")

    print(f"{'Модель':<20} | {'Формула':<30} | {'RMS':<8} | {'S'}")
    print("-" * 75)
    best_model = min(results, key=lambda t: t[2])

    for r in results:
        p_str = f" (r={r[4]:.3f})" if r[4] is not None else ""
        print(f"{r[0]:<20} | {r[1]:<30} | {r[2]:.4f} | {r[3]:.4f}{p_str}")

    print(f"\nНаилучшая аппроксимация: {best_model[0]} ({best_model[1]})")

    plt.legend()
    plt.grid(True)
    plt.title("Аппроксимация функций МНК")
    plt.show()


if __name__ == "__main__":
    main()