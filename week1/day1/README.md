# Day 1: Numpy Basics

## Dot Product

Dot product multiplies corresponding elements of two vectors and adds the results together. The two vectors must have the same length.

### Usage:

```python
dot([1, 2, 3], [4, 5, 6])
```

## Result:
```python
32
```


## Matrix Multiplication

Matrix multiplication combines two matrices by taking the dot product of rows from the first matrix with columns from the second. The number of columns in the first matrix must equal the number of rows in the second matrix.

### Usage:

```python
matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
```

### Result:

```python
[[19, 22], [43, 50]]
```


## Softmax

Softmax converts logits into probabilities. The resulting probabilities are between 0 and 1, and their sum is always 1.

### Usage:

```python
softmax([2, 1, 0])
```

### Result:

```python
[0.665, 0.245, 0.090]
```


## Cross Entropy

Entropy measures uncertainty in a probability distribution. Cross-entropy measures how well the predicted probabilities match the actual class. The lower the cross-entropy, the better the prediction.

### Usage:

```python
ce([0.7, 0.2, 0.1], 0)
```

### Result:

```python
0.35667
```
