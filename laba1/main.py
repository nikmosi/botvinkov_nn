from math import exp
from pprint import pprint


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def sigmoid_derivative(x: float) -> float:
    return (1 - x) * x


def main():
    memory = []
    memory.append([0, 1])
    memory.append([])
    memory.append([])
    memory.append([])
    sizes = [2, 3, 2, 1]
    network = [
        [
            [
                0.1,
                0.1,
                0.1,
            ],
            [
                0.5,
                0.5,
                0.5,
            ],
        ],
        [
            [
                0.3,
                0.3,
            ],
            [
                0.4,
                0.4,
            ],
            [
                0.5,
                0.5,
            ],
        ],
        [
            [
                0.1,
            ],
            [
                0.1,
            ],
        ],
    ]
    print("==================== before =========================")
    pprint(network, indent=2, width=20)
    print("==================== before =========================\n")

    print("==================== forward =========================")
    for step in range(len(network)):
        size = sizes[step + 1]
        size_prev = sizes[step]
        weights = network[step]
        mem_prev = memory[step]
        mem_next = [0] * size
        memory[step + 1] = mem_next
        for i in range(size):
            val_inp = 0
            for j in range(size_prev):
                val_inp += mem_prev[j] * weights[j][i]

            mem_next[i] = sigmoid(val_inp)
        memory[step - 1].extend(mem_next)

    pprint(memory, indent=2)
    print("==================== forward =========================\n")

    print("==================== backward =========================")
    o_ideal = 0.5
    o_actual = memory[len(sizes) - 1][0]
    d_o = (o_ideal - o_actual) * sigmoid_derivative(o_actual)
    deltas = [
        [d_o],
        [
            0,
            0,
        ],
        [
            0,
            0,
            0,
        ],
        [
            0,
            0,
        ],
    ]

    for step in range(1, len(deltas)):
        from_ = deltas[step - 1]
        to_ = deltas[step]
        mem = memory[-step - 1]
        weights = network[-step]
        for i in range(len(to_)):
            fn = sigmoid_derivative(mem[i])
            sum_ = 0
            for j, vf in enumerate(from_):
                sum_ += sum(weights[i]) * vf
            deltas[step][i] = fn * sum_

    pprint(network, indent=2, width=20)
    print("==================== backward =========================\n")


if __name__ == "__main__":
    main()
