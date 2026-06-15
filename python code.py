import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from tabulate import tabulate

# Function Evaluation Counter

class FunctionCounter:
    def __init__(self, f):
        self.f = f
        self.count = 0

    def __call__(self, x):
        self.count += 1
        return self.f(x)

    def reset(self):
        self.count = 0

# Trapezoidal Rule

def trapezoidal(f, a, b, n=1000):
    h = (b - a) / n

    result = 0.5 * (f(a) + f(b))

    for i in range(1, n):
        result += f(a + i * h)

    return result * h

# Composite Simpson Rule

def composite_simpson(f, a, b, n=1000):

    if n % 2 == 1:
        n += 1

    h = (b - a) / n

    result = f(a) + f(b)

    for i in range(1, n):
        x = a + i * h

        if i % 2 == 0:
            result += 2 * f(x)
        else:
            result += 4 * f(x)

    return result * h / 3


# Basic Simpson

def simpson(f, a, b):
    c = (a + b) / 2

    return ((b - a) / 6) * (
        f(a) +
        4 * f(c) +
        f(b)
    )


# Adaptive Simpson

accepted_intervals = []


def adaptive_simpson(
        f,
        a,
        b,
        eps,
        depth=0,
        max_depth=25):

    c = (a + b) / 2

    S = simpson(f, a, b)

    S_left = simpson(f, a, c)
    S_right = simpson(f, c, b)

    error_estimate = abs(S_left + S_right - S)

    if error_estimate < 15 * eps or depth >= max_depth:

        accepted_intervals.append((a, b))

        return (
            S_left +
            S_right +
            (S_left + S_right - S) / 15
        )

    return (
        adaptive_simpson(
            f,
            a,
            c,
            eps / 2,
            depth + 1,
            max_depth
        )
        +
        adaptive_simpson(
            f,
            c,
            b,
            eps / 2,
            depth + 1,
            max_depth
        )
    )


# Plot Adaptive Intervals

def plot_intervals(title):

    plt.figure(figsize=(12, 2))

    for a, b in accepted_intervals:
        plt.plot([a, b], [1, 1], linewidth=3)

    plt.yticks([])
    plt.title(title)

    plt.xlabel("x")

    plt.show()
    plt.savefig("Figure1.pdf", bbox_inches="tight")
    plt.savefig("Figure2.pdf", bbox_inches="tight")
    plt.savefig("Figure3.pdf", bbox_inches="tight")
    plt.savefig("Figure4.pdf", bbox_inches="tight")


# Test Function Runner

def run_test(name, func, a, b):

    global accepted_intervals

    accepted_intervals = []

    f = FunctionCounter(func)

    exact, _ = quad(func, a, b)

    f.reset()
    trap = trapezoidal(f, a, b)
    trap_calls = f.count

    f.reset()
    simp = composite_simpson(f, a, b)
    simp_calls = f.count

    f.reset()
    adaptive = adaptive_simpson(
        f,
        a,
        b,
        1e-8
    )
    adaptive_calls = f.count

    table = [
        [
            "Trapezoidal",
            trap,
            abs(trap - exact),
            trap_calls
        ],
        [
            "Composite Simpson",
            simp,
            abs(simp - exact),
            simp_calls
        ],
        [
            "Adaptive Simpson",
            adaptive,
            abs(adaptive - exact),
            adaptive_calls
        ]
    ]

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        tabulate(
            table,
            headers=[
                "Method",
                "Result",
                "Absolute Error",
                "Function Calls"
            ],
            tablefmt="grid"
        )
    )
    plot_intervals(
        f"Adaptive Subdivisions - {name}"
    )


# Test Cases

run_test(
    "Integral of sin(x)",
    lambda x: math.sin(x),
    0,
    math.pi
)
run_test(
    "Integral of exp(-x^2)",
    lambda x: math.exp(-x*x),
    0,
    1
)
run_test(
    "Integral of 1/(1+100x^2)",
    lambda x: 1/(1+100*x*x),
    0,
    1
)
run_test(
    "Integral of sqrt(x)",
    lambda x: math.sqrt(x),
    0,
    1
)