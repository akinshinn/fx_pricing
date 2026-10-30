import numpy as np 
import matplotlib.pyplot as plt 
from statistics import NormalDist


class CIR:
    def __init__(self, r0, kappa, theta, sigma, T, N, random_state=42):
        self.r0 = r0 # начальное значение
        self.kappa = kappa # скорость возврата к среднему
        self.theta = theta # среднее значение 
        self.sigma = sigma # коэффициент волатильности
        self.T = T # конечное время
        self.N = N # число шагов

        self.random_state = random_state
        np.random.seed(self.random_state)

        self.path = None 
        self.times = None

        self.paths = None 

    def get_next_time(self, t0):
        dt = self.T / self.N
        t1 = t0 + dt
        return t1

    def simulate_one_step(self, rt, t0):
        dt = self.get_next_time(t0) - t0
        Z = np.random.normal()
        rt_next = rt + self.kappa * (self.theta - rt) * dt + self.sigma * np.sqrt(rt) \
            * np.sqrt(dt) * Z
        return rt_next

    def simulate_path(self):
        path = [self.r0]
        times = [0]
        t0 = 0
        for _ in range(self.N):
            rt_next = self.simulate_one_step(path[-1], t0)
            path.append(rt_next)
            t0 = self.get_next_time(t0)
            times.append(t0)
            

        self.path = path 
        self.times = times

        return np.array(path)

    def simulate_multiple_paths(self, M):
        self.random_state = None 
        np.random.seed(self.random_state)
        paths = []
        for _ in range(M):
            path = self.simulate_path()
            paths.append(path)
        self.paths = np.array(paths)
        return paths

    def plot(self):
        plt.plot(self.times, self.path)
        plt.xlabel('Time')
        plt.ylabel('Interest Rate')
        plt.grid()
        plt.show()

    def plot_multiple_paths(self):
        for path in self.paths:
            plt.plot(self.times, path)
        plt.xlabel('Time')
        plt.ylabel('Interest Rate')
        plt.grid()
        plt.show()

    def get_monte_carlo_estimate(self, M, confidence=0.95):
        self.simulate_multiple_paths(M)

        # Строки — траектории, столбцы — моменты времени.
        paths = np.asarray(self.paths, dtype=float)

        mean = np.mean(paths, axis=0)
        median = np.median(paths, axis=0)
        variance = np.var(paths, axis=0, ddof=1)

        # Приближённый доверительный интервал для среднего по ЦПТ.
        alpha = 1 - confidence
        z = NormalDist().inv_cdf(1 - alpha / 2)
        standard_error = np.sqrt(variance / M)

        ci_lower = mean - z * standard_error
        ci_upper = mean + z * standard_error

        # Центральный квантильный интервал значений процесса.
        q_lower, q_upper = np.quantile(
            paths,
            [alpha / 2, 1 - alpha / 2],
            axis=0,
        )

        return {
            "times": np.asarray(self.times),
            "mean": mean,
            "median": median,
            "variance": variance,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "q_lower": q_lower,
            "q_upper": q_upper,
        }
