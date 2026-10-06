from week1.day1.numpy_basics import dot, matmul, softmax, ce


def test_dot():
    result = dot([1, 2, 3], [4, 5, 6])
    assert result == 32


def test_matmul():
    result = matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    assert result == [[19, 22], [43, 50]]


def test_softmax():
    result = softmax([2, 1, 0])
    assert result == [0.665, 0.245, 0.090]


def test_ce():
    result = ce([0.7, 0.2, 0.1], 0)
    assert result == 0.35667
