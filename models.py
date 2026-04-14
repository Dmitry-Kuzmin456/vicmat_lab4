import numpy as np


def solve_linear_system(matrix, vector):
    try:
        return np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return None


def get_rms(y_true, y_pred):
    n = len(y_true)
    s = np.sum((y_pred - y_true) ** 2)
    return np.sqrt(s / n), s


class ApproximationModels:
    @staticmethod
    def linear(x, y):
        n = len(x)
        matrix = np.array([[np.sum(x ** 2), np.sum(x)], [np.sum(x), n]])
        vector = np.array([np.sum(x * y), np.sum(y)])
        res = solve_linear_system(matrix, vector)
        if res is None: return None

        a, b = res
        func = lambda t: a * t + b

        # Коэффициент Пирсона
        mx, my = np.mean(x), np.mean(y)
        num = np.sum((x - mx) * (y - my))
        den = np.sqrt(np.sum((x - mx) ** 2) * np.sum((y - my) ** 2))
        pearson = num / den if den != 0 else 0

        return func, f"{a:.3f}x + {b:.3f}", pearson

    @staticmethod
    def poly_2(x, y):
        coeffs = np.polyfit(x, y, 2)
        func = np.poly1d(coeffs)
        return func, f"{coeffs[0]:.3f}x^2 + {coeffs[1]:.3f}x + {coeffs[2]:.3f}"

    @staticmethod
    def poly_3(x, y):
        coeffs = np.polyfit(x, y, 3)
        func = np.poly1d(coeffs)
        return func, f"{coeffs[0]:.3f}x^3 + {coeffs[1]:.3f}x^2 + {coeffs[2]:.3f}x + {coeffs[3]:.3f}"

    @staticmethod
    def exponential(x, y):
        # Авто-определение знака для логарифмирования
        sign = np.sign(np.mean(y))
        y_mod = y * sign
        y_mod[y_mod <= 0] = 1e-9  # Защита от log(0)

        coeffs = np.polyfit(x, np.log(y_mod), 1)
        a, b = np.exp(coeffs[1]), coeffs[0]
        return lambda t: sign * a * np.exp(b * t), f"{sign * a:.3f} * exp({b:.3f}x)"

    @staticmethod
    def log(x, y):
        x_mod = np.abs(x)
        x_mod[x_mod <= 0] = 1e-9
        coeffs = np.polyfit(np.log(x_mod), y, 1)
        return lambda t: coeffs[0] * np.log(np.abs(t) + 1e-9) + coeffs[1], \
            f"{coeffs[0]:.3f}ln|x| + {coeffs[1]:.3f}"

    @staticmethod
    def power(x, y):
        sign_y = np.sign(np.mean(y))
        x_mod = np.abs(x)
        y_mod = y * sign_y
        x_mod[x_mod <= 0] = 1e-9
        y_mod[y_mod <= 0] = 1e-9

        coeffs = np.polyfit(np.log(x_mod), np.log(y_mod), 1)
        a, b = np.exp(coeffs[1]), coeffs[0]
        return lambda t: sign_y * a * (np.abs(t) ** b), f"{sign_y * a:.3f} * |x|^{b:.3f}"