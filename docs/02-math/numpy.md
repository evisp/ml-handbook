# NumPy for Machine Learning: Efficient Array Operations

This tutorial teaches you to use NumPy, Python's fundamental library for numerical computing. You'll learn to create, manipulate, and perform operations on arrays efficiently - skills essential for every machine learning task.

**Estimated time:** 50 minutes

## Why This Matters

**Problem statement:** 

> Pure Python lists and loops are too slow for machine learning workloads involving millions of data points. 

Without NumPy's optimized array operations, training even simple models becomes impractically slow, and implementing ML algorithms from scratch is unnecessarily complex.

**Practical benefits:** NumPy provides extremely fast array operations through optimized C and Fortran libraries underneath. What takes seconds with Python loops completes in milliseconds with NumPy. 

> Every major ML library—scikit-learn, TensorFlow, PyTorch, pandas - is built on top of NumPy, making it the universal language of numerical computing in Python.

**Professional context:** NumPy is non-negotiable for ML work. Data preprocessing uses NumPy arrays, model inputs are NumPy arrays, predictions return as NumPy arrays. Understanding NumPy array operations, broadcasting, and vectorization enables you to write production-quality ML code and understand how libraries work under the hood.

## Core Concepts

### What is NumPy?

NumPy (Numerical Python) is the foundational library for scientific computing in Python. It provides a powerful `N`-dimensional array object and functions for fast operations on arrays.

**Why NumPy is fast:** NumPy operations are implemented in C and execute at compiled speeds, not interpreted Python speeds. 

> A NumPy operation on `1` million elements can be `100x` faster than an equivalent Python loop.

**Key insight:** The secret to NumPy's power is **vectorization**; applying operations to entire arrays at once instead of looping through elements. This shift in thinking from `loop over each element` to `operate on the whole array` is fundamental to efficient ML code.

### Arrays vs Lists

| Feature | Python List | NumPy Array |
|---------|-------------|-------------|
| Speed | Slow (interpreted) | Fast (compiled C) |
| Memory | More overhead | Compact, efficient |
| Data types | Mixed types allowed | Homogeneous (one type) |
| Operations | Manual loops needed | Vectorized operations |
| Use case | General programming | Numerical computing |

**When to use NumPy:** Anytime you're working with numerical data, especially large datasets, mathematical operations, or preparing data for ML models.

## Step-by-Step Instructions

### 1.  Installing and Importing NumPy

NumPy must be installed and imported before use.

**Install NumPy:**

```bash
pip install numpy
```

**Import NumPy (standard convention):**

```python
import numpy as np
```

**Why `np`:** The alias `np` is universally used in the Python data science community, making code readable and recognizable across projects.

**Verify installation:**

```python
import numpy as np
print(np.__version__)
```

**Expected output:**

```
1.24.3 # or a similar version
```

### 2. Creating NumPy Arrays from Lists

The simplest way to create arrays is by converting Python lists.

**Create a 1D array (vector):**

```python
# From a simple list
my_list = [1, 2, 3, 4, 5]
vector = np.array(my_list)

print(vector)
print(f"Type: {type(vector)}")
print(f"Shape: {vector.shape}")
```

**Expected output:**

```
[1 2 3 4 5]
Type: lass 'numpy.ndarrayay'>
Shape: (5,)
```

**Why shape matters:** The shape `(5,)` indicates a 1D array with 5 elements. Understanding shapes is critical for ensuring array operations are compatible.

**Create a 2D array (matrix):**

```python
# From a nested list
my_matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

matrix = np.array(my_matrix)
print(matrix)
print(f"Shape: {matrix.shape}")
```

**Expected output:**

```
[[1 2 3]
 [4 5 6]
 [7 8 9]]
Shape: (3, 3)
```

**Interpretation:** A shape of `(3, 3)` means 3 rows and 3 columns. In ML terminology, this could represent 3 data samples with 3 features each.

### 3. Built-in Array Creation Functions

NumPy provides convenient functions for generating arrays without manually typing values.

#### arange: Evenly Spaced Values

Similar to Python's `range()`, but returns a NumPy array.

```python
# Create array from 0 to 10 (exclusive)
arr = np.arange(0, 11)
print(arr)

# With step size
arr_step = np.arange(0, 11, 2)
print(arr_step)

# Start from non-zero
arr_tens = np.arange(10, 101, 10)
print(arr_tens)
```

**Expected output:**

```
[ 0  1  2  3  4  5  6  7  8  9 10]
[ 0  2  4  6  8 10]
[ 10  20  30  40  50  60  70  80  90 100]
```

**ML use case:** Creating indices, generating sequences for time series data, or creating evenly spaced feature bins.

#### zeros and ones: Initialize Arrays

Useful for creating placeholder arrays before filling them with computed values.

```python
# 1D array of zeros
zeros_1d = np.zeros(5)
print(zeros_1d)

# 2D array of zeros
zeros_2d = np.zeros((3, 4))
print(zeros_2d)

# 2D array of ones
ones_2d = np.ones((2, 3))
print(ones_2d)
```

**Expected output:**

```
[0. 0. 0. 0. 0.]
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
[[1. 1. 1.]
 [1. 1. 1.]]
```

**Why initialize with zeros:** Many ML algorithms accumulate results (like gradient sums in backpropagation) starting from zero arrays.

#### linspace: Linearly Spaced Values

Unlike `arange` which uses step size, `linspace` specifies how many values you want.

```python
# 10 evenly spaced values between 0 and 100
lin = np.linspace(0, 100, 10)
print(lin)

# 5 values between 2 and 20
lin2 = np.linspace(2, 20, 5)
print(lin2)
```

**Expected output:**

```
[  0.          11.11111111  22.22222222  33.33333333  44.44444444
  55.55555556  66.66666667  77.77777778  88.88888889 100.        ]
[ 2.   6.5 11.  15.5 20. ]
```

**ML use case:** Creating evenly spaced feature values for plotting decision boundaries or generating test inputs.

#### eye: Identity Matrix

An identity matrix has ones on the diagonal and zeros elsewhere. Critical for linear algebra operations.

```python
# 3x3 identity matrix
identity = np.eye(3)
print(identity)

# 5x5 identity matrix
identity_5 = np.eye(5)
print(identity_5)
```

**Expected output:**

```
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
```

**Why identity matrices matter:** Used in matrix inversion, solving linear systems, and initializing certain neural network layers.

### 4. Random Number Generation

Random numbers are essential for initializing model weights, creating synthetic data, and splitting datasets.

#### rand: Uniform Random Values

Generates random numbers uniformly distributed between 0 and 1.

```python
# 1D array of 5 random values
rand_1d = np.random.rand(5)
print(rand_1d)

# 3x3 matrix of random values
rand_2d = np.random.rand(3, 3)
print(rand_2d)
```

**Expected output (values will vary):**

```
[0.52134728 0.76543218 0.23498765 0.98123456 0.12345678]
[[0.41234567 0.87654321 0.23456789]
 [0.65432109 0.34567890 0.78901234]
 [0.12345678 0.56789012 0.90123456]]
```

**ML use case:** Initializing neural network weights with small random values to break symmetry.

#### randn: Standard Normal Distribution

Generates random numbers from a Gaussian (normal) distribution with mean 0 and standard deviation 1.

```python
# 2x3 matrix from standard normal distribution
randn_2d = np.random.randn(2, 3)
print(randn_2d)
```

**Expected output (values will vary):**

```
[[ 0.51234567 -1.23456789  0.87654321]
 [-0.34567890  1.45678901 -0.67890123]]
```

**Why normal distribution:** Many ML algorithms assume data follows a normal distribution. Random normal values are commonly used for weight initialization in deep learning.

#### randint: Random Integers

Generates random integers within a specified range.

```python
# 10 random integers between 1 and 100 (exclusive)
rand_ints = np.random.randint(1, 100, 10)
print(rand_ints)

# Single random integer
single = np.random.randint(1, 100)
print(single)
```

**Expected output (values will vary):**

```
[42 73 18 91 35 67 22 88 14 59]
76
```

**ML use case:** Creating random indices for data shuffling, generating discrete synthetic data, or random sampling.

### 5. Array Attributes and Methods

Understanding array properties helps you debug shape mismatches and verify data dimensions.

**Key attributes:**

```python
arr = np.arange(25)
print(f"Array: {arr}")
print(f"Shape: {arr.shape}")
print(f"Size (total elements): {arr.size}")
print(f"Data type: {arr.dtype}")
print(f"Number of dimensions: {arr.ndim}")
```

**Expected output:**

```
Array: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
Shape: (25,)
Size (total elements): 25
Data type: int64
Number of dimensions: 1
```

### 6. Reshaping Arrays

Reshaping changes array dimensions without changing the data. Essential for preparing data for ML models.

**Reshape a 1D array to 2D:**

```python
arr = np.arange(30)
print("Original shape:", arr.shape)

# Reshape to 5 rows, 6 columns
reshaped = arr.reshape(5, 6)
print("\nReshaped to (5, 6):")
print(reshaped)
print("New shape:", reshaped.shape)
```

**Expected output:**

```
Original shape: (30,)

Reshaped to (5, 6):
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]
 [24 25 26 27 28 29]]
New shape: (5, 6)
```

**Important rule:** Total elements must remain the same. You cannot reshape 30 elements into a `(4, 8)` array because $$4 \times 8 = 32 \neq 30$$.

**ML use case:** Image data often needs reshaping from flat vectors to $$(height, width, channels)$$ format for CNNs.

### 7. Finding Maximum, Minimum, and Indices

These methods help identify extreme values in data.

```python
ranarr = np.random.randint(0, 100, 10)
print("Array:", ranarr)
print("Maximum value:", ranarr.max())
print("Minimum value:", ranarr.min())
print("Index of maximum:", ranarr.argmax())
print("Index of minimum:", ranarr.argmin())
```

**Expected output (values will vary):**

```
Array: [42 87 15 93 28 61 74 19 56 38]
Maximum value: 93
Minimum value: 15
Index of maximum: 3
Index of minimum: 2
```

**ML use case:** Finding the predicted class in classification (index of maximum probability), identifying outliers, or locating peak activations in neural networks.

### 8. Array Indexing and Slicing

NumPy indexing works similarly to Python lists but extends to multiple dimensions.

#### Basic 1D Indexing

```python
arr = np.arange(0, 11)
print("Array:", arr)

# Access single element
print("Element at index 5:", arr[5])

# Slice a range
print("Elements 1-5:", arr[1:6])

# Slice from start
print("First 4 elements:", arr[:4])

# Slice to end
print("Last 3 elements:", arr[-3:])
```

**Expected output:**

```
Array: [ 0  1  2  3  4  5  6  7  8  9 10]
Element at index 5: 5
Elements 1-5: [1 2 3 4 5]
First 4 elements: [0 1 2 3]
Last 3 elements: [ 8  9 10]
```

#### Broadcasting: Assigning Values to Slices

NumPy allows assigning a single value to multiple array elements at once.

```python
arr = np.arange(0, 11)
print("Original:", arr)

# Set indices 3-6 to 300
arr[3:6] = 300
print("After broadcasting:", arr)
```

**Expected output:**

```
Original: [ 0  1  2  3  4  5  6  7  8  9 10]
After broadcasting: [  0   1   2 300 300 300   6   7   8   9  10]
```

**Why broadcasting matters:** Enables efficient operations on entire array slices without loops, a cornerstone of vectorized programming.

### 9. 2D Array Indexing

Matrices require row and column indices.

**Access elements and slices:**

```python
arr_2d = np.array([[5, 10, 15],
                   [20, 25, 30],
                   [35, 40, 45]])

print("Full matrix:")
print(arr_2d)

# Access single element (row 1, column 2)
print("\nElement [1, 2]:", arr_2d[1, 2])

# Get entire row
print("\nRow 2:", arr_2d[2])

# Get entire column
print("\nColumn 1:", arr_2d[:, 1])

# Slice submatrix (top-right 2x2)
print("\nTop-right 2x2:")
print(arr_2d[0:2, 1:3])

# All rows, columns 1 and 2
print("\nAll rows, columns 1-2:")
print(arr_2d[:, 1:3])
```

**Expected output:**

```
Full matrix:
[[ 5 10 15]
 [20 25 30]
 [35 40 45]]

Element [1, 2]: 30

Row 2: [35 40 45]

Column 1: [10 25 40]

Top-right 2x2:
[[10 15]
 [25 30]]

All rows, columns 1-2:
[[10 15]
 [25 30]
 [40 45]]
```

**Notation:** `arr_2d[row, col]` is clearer than `arr_2d[row][col]` and slightly faster.

### 10. Boolean Indexing (Conditional Selection)

Select array elements that meet a condition—extremely powerful for data filtering.

```python
arr = np.arange(1, 11)
print("Array:", arr)

# Boolean mask: which elements are greater than 5?
mask = arr > 5
print("\nBoolean mask (arr > 5):", mask)

# Select only elements greater than 5
filtered = arr[arr > 5]
print("Elements > 5:", filtered)

# Select elements less than or equal to 5
filtered_le = arr[arr <= 5]
print("Elements <= 5:", filtered_le)
```

**Expected output:**

```
Array: [ 1  2  3  4  5  6  7  8  9 10]

Boolean mask (arr > 5): [False False False False False  True  True  True  True  True]
Elements > 5: [ 6  7  8  9 10]
Elements <= 5: [1 2 3 4 5]
```

**How it works:** The condition `arr > 5` returns a Boolean array. Using this as an index returns only `True` positions.

**ML use case:** Filtering outliers, selecting samples meeting criteria, or applying threshold-based decisions.

### 11. Vectorized Arithmetic Operations

NumPy performs element-wise operations across entire arrays without loops.

```python
arr = np.arange(0, 10)
print("Array:", arr)

# Array + Array (element-wise)
print("arr + arr:", arr + arr)

# Array * Array (element-wise)
print("arr * arr:", arr * arr)

# Array - Array
print("arr - arr:", arr - arr)

# Scalar operations
print("arr + 10:", arr + 10)
print("arr * 2:", arr * 2)
print("arr ** 2:", arr ** 2)
```

**Expected output:**

```
Array: [0 1 2 3 4 5 6 7 8 9]
arr + arr: [ 0  2  4  6  8 10 12 14 16 18]
arr * arr: [ 0  1  4  9 16 25 36 49 64 81]
arr - arr: [0 0 0 0 0 0 0 0 0 0]
arr + 10: [10 11 12 13 14 15 16 17 18 19]
arr * 2: [ 0  2  4  6  8 10 12 14 16 18]
arr ** 2: [ 0  1  4  9 16 25 36 49 64 81]
```

**Critical insight:** Operations like `arr * arr` multiply corresponding elements, not matrix multiplication. For matrix multiplication, use `np.dot()` or `@` operator.

### 12. Universal Functions (ufuncs)

NumPy provides optimized mathematical functions that operate element-wise.

```python
arr = np.arange(1, 11)

# Square root
print("Square roots:", np.sqrt(arr))

# Exponential
print("Exponentials:", np.exp(arr[:3]))  # First 3 to avoid huge numbers

# Logarithm
print("Natural log:", np.log(arr))

# Trigonometric
angles = np.array([0, np.pi/2, np.pi])
print("Sine:", np.sin(angles))
```

**Expected output:**

```
Square roots: [1.         1.41421356 1.73205081 2.         2.23606798 2.44948975
 2.64575131 2.82842712 3.         3.16227766]
Exponentials: [ 2.71828183  7.3890561  20.08553692]
Natural log: [0.         0.69314718 1.09861229 1.38629436 1.60943791 1.79175947
 1.94591015 2.07944154 2.19722458 2.30258509]
Sine: [0.0000000e+00 1.0000000e+00 1.2246468e-16]
```

**ML use case:** Activation functions (sigmoid, tanh), normalizing data (log transform), computing distances (square root of sum of squares).

## Quick Reference

| Operation | Code | Use Case |
|-----------|------|----------|
| Create from list | `np.array([1, 2, 3])` | Convert Python data to NumPy |
| Range | `np.arange(0, 10, 2)` | Sequential values |
| Evenly spaced | `np.linspace(0, 1, 100)` | Smooth intervals |
| Zeros/Ones | `np.zeros((3, 4))`, `np.ones((2, 3))` | Initialize arrays |
| Random uniform | `np.random.rand(3, 3)` | Random weights (0-1) |
| Random normal | `np.random.randn(3, 3)` | Gaussian initialization |
| Random integers | `np.random.randint(1, 100, 10)` | Discrete sampling |
| Reshape | `arr.reshape(5, 6)` | Change dimensions |
| Indexing | `arr`, `arr[1:5]`, `arr[arr > 3]` | Access elements |
| 2D indexing | `matrix[row, col]`, `matrix[:, col]` | Access matrix data |
| Element-wise ops | `arr + 5`, `arr * arr`, `arr ** 2` | Vectorized arithmetic |
| Math functions | `np.sqrt()`, `np.exp()`, `np.log()` | Element-wise functions |

## Summary & Next Steps

**Key accomplishments:** You've learned to create NumPy arrays using multiple methods, manipulate array shapes and dimensions, perform efficient indexing and slicing including Boolean selection, and apply vectorized operations that replace slow Python loops.

**Best practices:**

- **Always use NumPy arrays** for numerical data instead of lists
- **Think in terms of array operations** rather than element-by-element loops
- **Check array shapes frequently** when debugging to catch dimension mismatches early
- **Use vectorization** wherever possible for speed and code clarity
- **Understand broadcasting** to efficiently combine arrays of different shapes

**Performance mindset:** If you find yourself writing a `for` loop to process array elements, ask: "Can I vectorize this?" The answer is almost always yes in NumPy.

**Connections to ML:**

- **Data representation:** Datasets become NumPy arrays with shape `(samples, features)`
- **Model operations:** Linear layers compute $$\mathbf{y} = W\mathbf{x} + \mathbf{b}$$ using NumPy operations
- **Gradient descent:** Parameter updates use vectorized addition and multiplication
- **Image processing:** Images are NumPy arrays with shape `(height, width, channels)`

**External resources:**

- [NumPy Official Documentation](https://numpy.org/doc/stable/) - comprehensive reference and tutorials
- [NumPy for Absolute Beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) - official beginner guide
- [From Python to NumPy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/) - free online book on vectorization
