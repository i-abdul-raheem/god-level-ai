from math import exp, log


class DimensionMismatch(Exception):
    pass


def dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise DimensionMismatch(f"Dimension mismatch: {len(a)} != {len(b)}")

    try:
        return sum(x * y for x, y in zip(a, b))
    except TypeError as e:
        raise ValueError(f"Invalid vector: {e}")


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    if not a or not b or not a[0] or not b[0]:
        raise ValueError("Matrices cannot be empty")

    if len(a[0]) != len(b):
        raise DimensionMismatch(
            f"Dimension mismatch: {len(a)}x{len(a[0])} != {len(b)}x{len(b[0])}"
        )

    try:
        b_transpose = [list(row) for row in zip(*b)]
        return [[dot(row, col) for col in b_transpose] for row in a]
    except (TypeError, IndexError) as e:
        raise ValueError(f"Invalid matrix: {e}")


def softmax(logits: list[float]) -> list[float]:
    try:
        probs = [exp(x) for x in logits]
        total = sum(probs)
        return [round(p / total, 3) for p in probs]
    except (TypeError, OverflowError, ZeroDivisionError) as e:
        raise ValueError(f"Invalid logits: {e}")


def ce(probs: list[float], target: int) -> float:
    if target < 0 or target >= len(probs):
        raise ValueError("Target is out of bounds")
    return round(-log(probs[target]), 5)
