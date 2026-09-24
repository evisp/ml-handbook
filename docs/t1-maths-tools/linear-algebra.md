# Linear Algebra with NumPy

<span class="badge badge--time">75 min</span>
<span class="badge badge--level">Foundations</span>
<span class="badge">Needs: Python running</span>

Every dataset you will ever work with is a grid of numbers. Every model is a
sequence of operations on that grid. This page covers those operations, first by
hand so you know what they are, then in NumPy so you can run them on real data.

## Why this matters

Linear algebra is not a hurdle placed before machine learning. It is the
notation machine learning is written in. When a paper says a layer computes
$Wx + b$, that is a matrix multiplied by a vector plus another vector, and you
will be able to read it by the end of this page.

There is a more immediate reason. The most common error you will hit this year
is a shape mismatch: two arrays that cannot be combined because their dimensions
do not line up. Understanding shapes is what turns that from a mystery into a
thirty second fix.

## Your data is a matrix

Before any notation, the picture that matters.

<svg viewBox="0 0 680 262" role="img" aria-labelledby="dm-title dm-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="dm-title">A dataset as a matrix</title>
<desc id="dm-desc">A table of four customers and four features. Each row is one customer and is a vector of four numbers. Each column is one feature across all customers.</desc>
<text x="60" y="30" font-size="12" font-weight="700" fill="var(--h-graphite)">a dataset of 4 customers and 4 features, shape (4, 4)</text>
<rect x="60" y="48" width="110" height="32" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="170" y="48" width="110" height="32" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="280" y="48" width="110" height="32" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="390" y="48" width="110" height="32" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="115" y="69" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">age</text>
<text x="225" y="69" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">income</text>
<text x="335" y="69" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">visits</text>
<text x="445" y="69" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">spend</text>
<rect x="60" y="84" width="440" height="32" fill="none" stroke="var(--h-surface-line)"/>
<text x="115" y="105" text-anchor="middle" font-size="12" fill="var(--h-graphite)">34</text>
<text x="225" y="105" text-anchor="middle" font-size="12" fill="var(--h-graphite)">41200</text>
<text x="335" y="105" text-anchor="middle" font-size="12" fill="var(--h-graphite)">12</text>
<text x="445" y="105" text-anchor="middle" font-size="12" fill="var(--h-graphite)">890</text>
<rect x="60" y="118" width="440" height="32" fill="var(--h-cherry)"/>
<text x="115" y="139" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">28</text>
<text x="225" y="139" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">33800</text>
<text x="335" y="139" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">5</text>
<text x="445" y="139" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">410</text>
<rect x="60" y="152" width="440" height="32" fill="none" stroke="var(--h-surface-line)"/>
<text x="115" y="173" text-anchor="middle" font-size="12" fill="var(--h-graphite)">51</text>
<text x="225" y="173" text-anchor="middle" font-size="12" fill="var(--h-graphite)">67500</text>
<text x="335" y="173" text-anchor="middle" font-size="12" fill="var(--h-graphite)">22</text>
<text x="445" y="173" text-anchor="middle" font-size="12" fill="var(--h-graphite)">2150</text>
<rect x="60" y="186" width="440" height="32" fill="none" stroke="var(--h-surface-line)"/>
<text x="115" y="207" text-anchor="middle" font-size="12" fill="var(--h-graphite)">43</text>
<text x="225" y="207" text-anchor="middle" font-size="12" fill="var(--h-graphite)">52000</text>
<text x="335" y="207" text-anchor="middle" font-size="12" fill="var(--h-graphite)">9</text>
<text x="445" y="207" text-anchor="middle" font-size="12" fill="var(--h-graphite)">1240</text>
<line x1="512" y1="134" x2="504" y2="134" stroke="var(--h-cherry)" stroke-width="2"/>
<text x="518" y="126" font-size="11.5" font-weight="700" fill="var(--h-cherry)">one row is one customer,</text>
<text x="518" y="142" font-size="11.5" font-weight="700" fill="var(--h-cherry)">a vector of 4 numbers</text>
<text x="60" y="240" font-size="12" fill="var(--h-graphite)">rows are samples, columns are features. Every dataset you meet arrives in this shape.</text>
</svg>

One row is a vector. The whole table is a matrix. That is the entire mapping
between the maths and your data, and everything below is operations on those two
objects.

## Vectors

A vector is an ordered list of numbers. Three ways of seeing the same thing:

- **A list**: `[3, 1]`. This is the view you will type.
- **An arrow**: pointing from the origin out to that position.
- **A point**: a location in space with one coordinate per number.

A customer with age, income, visits and spend is a point in four dimensional
space. You cannot picture four dimensions, and you do not need to. Everything
that works in two dimensions works identically in four hundred.

### The operations

<svg viewBox="0 0 680 300" role="img" aria-labelledby="vo-title vo-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="vo-title">Vector addition and scaling on a grid</title>
<desc id="vo-desc">Two vectors drawn from the origin. Their sum is the diagonal of the parallelogram they form. A vector scaled by two keeps its direction and doubles its length.</desc>
<defs><marker id="va-steel" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-steel)"/></marker><marker id="va-cherry" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-cherry)"/></marker></defs>
<line x1="60" y1="40" x2="60" y2="240" stroke="var(--h-surface-line)"/>
<line x1="100" y1="40" x2="100" y2="240" stroke="var(--h-surface-line)"/>
<line x1="140" y1="40" x2="140" y2="240" stroke="var(--h-surface-line)"/>
<line x1="180" y1="40" x2="180" y2="240" stroke="var(--h-surface-line)"/>
<line x1="220" y1="40" x2="220" y2="240" stroke="var(--h-surface-line)"/>
<line x1="260" y1="40" x2="260" y2="240" stroke="var(--h-surface-line)"/>
<line x1="300" y1="40" x2="300" y2="240" stroke="var(--h-surface-line)"/>
<line x1="340" y1="40" x2="340" y2="240" stroke="var(--h-surface-line)"/>
<line x1="380" y1="40" x2="380" y2="240" stroke="var(--h-surface-line)"/>
<line x1="420" y1="40" x2="420" y2="240" stroke="var(--h-surface-line)"/>
<line x1="60" y1="240" x2="420" y2="240" stroke="var(--h-surface-line)"/>
<line x1="60" y1="200" x2="420" y2="200" stroke="var(--h-surface-line)"/>
<line x1="60" y1="160" x2="420" y2="160" stroke="var(--h-surface-line)"/>
<line x1="60" y1="120" x2="420" y2="120" stroke="var(--h-surface-line)"/>
<line x1="60" y1="80" x2="420" y2="80" stroke="var(--h-surface-line)"/>
<line x1="60" y1="40" x2="420" y2="40" stroke="var(--h-surface-line)"/>
<line x1="180" y1="200" x2="220" y2="80" stroke="var(--h-steel)" stroke-width="1.5" stroke-dasharray="4 4"/>
<line x1="100" y1="120" x2="220" y2="80" stroke="var(--h-steel)" stroke-width="1.5" stroke-dasharray="4 4"/>
<line x1="60" y1="240" x2="300" y2="160" stroke="var(--h-cherry)" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#va-cherry)" opacity="0.55"/>
<line x1="60" y1="240" x2="180" y2="200" stroke="var(--h-steel)" stroke-width="2.5" marker-end="url(#va-steel)"/>
<line x1="60" y1="240" x2="100" y2="120" stroke="var(--h-steel)" stroke-width="2.5" marker-end="url(#va-steel)"/>
<line x1="60" y1="240" x2="220" y2="80" stroke="var(--h-cherry)" stroke-width="3" marker-end="url(#va-cherry)"/>
<text x="188" y="216" font-size="12.5" font-weight="700" fill="var(--h-graphite)">v1 = [3, 1]</text>
<text x="62" y="112" font-size="12.5" font-weight="700" fill="var(--h-graphite)">v2 = [1, 3]</text>
<text x="230" y="74" font-size="12.5" font-weight="700" fill="var(--h-cherry)">v1 + v2 = [4, 4]</text>
<text x="292" y="152" font-size="11.5" font-weight="700" fill="var(--h-cherry)" opacity="0.8">2 * v1 = [6, 2]</text>
<text x="460" y="100" font-size="12.5" fill="var(--h-graphite)">Adding puts one vector</text>
<text x="460" y="118" font-size="12.5" fill="var(--h-graphite)">nose to tail with the other.</text>
<text x="460" y="136" font-size="12.5" fill="var(--h-graphite)">The sum is the diagonal</text>
<text x="460" y="154" font-size="12.5" fill="var(--h-graphite)">of the parallelogram.</text>
<text x="460" y="188" font-size="12.5" fill="var(--h-graphite)">Scaling keeps the direction</text>
<text x="460" y="206" font-size="12.5" fill="var(--h-graphite)">and changes the length.</text>
<text x="460" y="224" font-size="12.5" fill="var(--h-graphite)">A negative scalar flips it.</text>
<text x="60" y="278" font-size="12" fill="var(--h-graphite)">Both operations work the same way in 2 dimensions and in 400. Only the picture stops working.</text>
</svg>

Addition and scaling are element by element. Nothing surprising happens.

```python
def add_vectors(v1, v2):
    """Add two vectors element by element."""
    if len(v1) != len(v2):
        return None
    return [v1[i] + v2[i] for i in range(len(v1))]


def subtract_vectors(v1, v2):
    """Subtract v2 from v1 element by element."""
    if len(v1) != len(v2):
        return None
    return [v1[i] - v2[i] for i in range(len(v1))]


def scalar_multiply(c, v):
    """Multiply every element of v by the number c."""
    return [c * x for x in v]


print(add_vectors([1, 2, 3], [4, 5, 6]))
print(subtract_vectors([10, 8, 6], [1, 2, 3]))
print(scalar_multiply(2, [1, 2, 3]))
```

```
[5, 7, 9]
[9, 6, 3]
[2, 4, 6]
```

Returning `None` on a length mismatch is a deliberate choice. Adding vectors of
different lengths is not a smaller answer, it is a question that does not mean
anything.

**Where you meet these.** Gradient descent updates parameters by adding a scaled
gradient vector to the current one. That is these two operations, run a few
thousand times.

### The dot product

Multiply matching elements and add the results. One number comes out.

$$
v_1 \cdot v_2 = \sum_i v_{1i} \, v_{2i}
$$

```python
def dot_product(v1, v2):
    """Sum of the products of matching elements."""
    if len(v1) != len(v2):
        return None
    return sum(v1[i] * v2[i] for i in range(len(v1)))


print(dot_product([1, 2, 3], [4, 5, 6]))
```

```
32
```

That is $1 \times 4 + 2 \times 5 + 3 \times 6$.

The arithmetic is trivial. What makes it the most important operation on this
page is what the number means. The dot product also equals

$$
v_1 \cdot v_2 = |v_1| \, |v_2| \cos \theta
$$

where $\theta$ is the angle between them. So its sign and size tell you how much
two vectors point the same way.

<svg viewBox="0 0 680 236" role="img" aria-labelledby="dp-title dp-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="dp-title">What the sign of the dot product means</title>
<desc id="dp-desc">Three pairs of vectors. Pointing the same way gives a large positive dot product. At right angles gives zero. Pointing in opposite directions gives a negative dot product.</desc>
<defs><marker id="dp-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--h-cherry)"/></marker></defs>
<rect x="5" y="20" width="210" height="150" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="110" y="44" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">same direction</text>
<line x1="60" y1="140" x2="165" y2="100" stroke="var(--h-cherry)" stroke-width="2.5" marker-end="url(#dp-arrow)"/>
<line x1="60" y1="140" x2="150" y2="72" stroke="var(--h-cherry)" stroke-width="2.5" marker-end="url(#dp-arrow)"/>
<text x="110" y="192" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">large positive</text>
<rect x="235" y="20" width="210" height="150" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="340" y="44" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">at right angles</text>
<line x1="290" y1="140" x2="395" y2="140" stroke="var(--h-cherry)" stroke-width="2.5" marker-end="url(#dp-arrow)"/>
<line x1="290" y1="140" x2="290" y2="66" stroke="var(--h-cherry)" stroke-width="2.5" marker-end="url(#dp-arrow)"/>
<text x="340" y="192" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">zero</text>
<rect x="465" y="20" width="210" height="150" rx="12" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="570" y="44" text-anchor="middle" font-size="12" font-weight="700" fill="var(--md-default-fg-color)">opposite directions</text>
<line x1="570" y1="105" x2="660" y2="72" stroke="var(--h-cherry)" stroke-width="2.5" marker-end="url(#dp-arrow)"/>
<line x1="570" y1="105" x2="484" y2="140" stroke="var(--h-cherry)" stroke-width="2.5" marker-end="url(#dp-arrow)"/>
<text x="570" y="192" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--h-cherry)">negative</text>
<text x="340" y="222" text-anchor="middle" font-size="12" fill="var(--h-graphite)">the dot product measures how much two vectors agree</text>
</svg>

!!! ml "ML connection"

    In trimester three you will turn sentences into vectors called embeddings,
    and then find which two sentences mean similar things. The way you do it is
    the dot product, with the lengths divided out. Every "find similar" feature
    you have ever used rests on this operation.

    A single neuron also computes a dot product between its inputs and its
    weights. There is no third idea hiding underneath.

## Matrices

A matrix is a rectangular grid of numbers. An $m \times n$ matrix has $m$ rows
and $n$ columns, and rows always come first.

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
$$

That is a $2 \times 3$ matrix. Element $A_{ij}$ sits at row $i$, column $j$.

In Python without libraries, it is a list of lists.

```python
matrix = [[1, 2, 3],
          [4, 5, 6]]

rows = len(matrix)
cols = len(matrix[0])
print(f"shape: {rows} x {cols}")
print(f"element at row 1, column 2: {matrix[1][2]}")
```

```
shape: 2 x 3
element at row 1, column 2: 6
```

### Transpose

The transpose flips a matrix over its diagonal. Rows become columns.

```python
def transpose(matrix):
    """Swap rows and columns."""
    rows, cols = len(matrix), len(matrix[0])
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]


print(transpose([[1, 2, 3], [4, 5, 6]]))
```

```
[[1, 4], [2, 5], [3, 6]]
```

A $2 \times 3$ becomes a $3 \times 2$. Written $A^T$. You will mostly use it to
make two shapes line up so a multiplication is allowed.

### Matrix multiplication

This is the operation everything else is built from, and it is not element by
element.

To multiply $A$ by $B$, the number of columns in $A$ must equal the number of
rows in $B$. Element $C_{ij}$ is the dot product of row $i$ of $A$ with column
$j$ of $B$.

<svg viewBox="0 0 680 212" role="img" aria-labelledby="mm-title mm-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="mm-title">The shape rule for matrix multiplication</title>
<desc id="mm-desc">A two by three matrix times a three by two matrix gives a two by two result. The inner dimensions, both three, must match. The outer dimensions become the shape of the result.</desc>
<text x="340" y="14" text-anchor="middle" font-size="11.5" font-weight="600" fill="var(--h-graphite)">the outer numbers become the shape of the result</text>
<polyline points="95,54 95,32 495,32 495,54" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<polyline points="335,54 335,22 535,22 535,54" fill="none" stroke="var(--h-steel)" stroke-width="1.5"/>
<rect x="40" y="60" width="150" height="86" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="115" y="92" text-anchor="middle" font-size="15" font-weight="700" fill="var(--md-default-fg-color)">A</text>
<text x="95" y="126" text-anchor="middle" font-size="17" font-weight="700" fill="var(--md-default-fg-color)">2</text>
<text x="115" y="126" text-anchor="middle" font-size="15" fill="var(--h-graphite)">x</text>
<text x="135" y="126" text-anchor="middle" font-size="17" font-weight="700" fill="var(--h-cherry)">3</text>
<text x="215" y="110" text-anchor="middle" font-size="18" fill="var(--h-graphite)">x</text>
<rect x="240" y="60" width="150" height="86" rx="10" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="315" y="92" text-anchor="middle" font-size="15" font-weight="700" fill="var(--md-default-fg-color)">B</text>
<text x="295" y="126" text-anchor="middle" font-size="17" font-weight="700" fill="var(--h-cherry)">3</text>
<text x="315" y="126" text-anchor="middle" font-size="15" fill="var(--h-graphite)">x</text>
<text x="335" y="126" text-anchor="middle" font-size="17" font-weight="700" fill="var(--md-default-fg-color)">2</text>
<text x="415" y="110" text-anchor="middle" font-size="18" fill="var(--h-graphite)">=</text>
<rect x="440" y="60" width="150" height="86" rx="10" fill="var(--h-cherry)"/>
<text x="515" y="92" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">A B</text>
<text x="495" y="126" text-anchor="middle" font-size="17" font-weight="700" fill="#ffffff">2</text>
<text x="515" y="126" text-anchor="middle" font-size="15" fill="#ffffff" opacity="0.85">x</text>
<text x="535" y="126" text-anchor="middle" font-size="17" font-weight="700" fill="#ffffff">2</text>
<polyline points="135,152 135,176 295,176 295,152" fill="none" stroke="var(--h-cherry)" stroke-width="2"/>
<text x="215" y="198" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)">these must match, or the operation is not defined</text>
</svg>

```python
def matrix_multiply(A, B):
    """Multiply A by B. Returns None if the shapes do not allow it."""
    rows_a, cols_a = len(A), len(A[0])
    rows_b, cols_b = len(B), len(B[0])
    if cols_a != rows_b:
        print(f"cannot multiply {rows_a}x{cols_a} by {rows_b}x{cols_b}")
        return None
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            total = 0
            for k in range(cols_a):
                total += A[i][k] * B[k][j]
            result[i][j] = total
    return result


A = [[1, 2, 3],
     [4, 5, 6]]
B = [[7, 8],
     [9, 10],
     [11, 12]]

for row in matrix_multiply(A, B):
    print(row)
```

```
[58, 64]
[139, 154]
```

Where 58 comes from: row one of $A$ is `[1, 2, 3]`, column one of $B$ is
`[7, 9, 11]`, and $1 \times 7 + 2 \times 9 + 3 \times 11 = 58$.

Write that function once. It is the only time you will do this by hand, and it
is worth knowing that three nested loops is what the one character `@` is hiding.

!!! ml "ML connection"

    A layer of a neural network computes $Wx + b$. That is this operation plus a
    vector addition. A network with twenty layers is this operation twenty
    times, with a simple function applied in between.

## Now do it properly

Here is the problem with everything above. Run those loops on a real dataset and
you will wait.

```python
import numpy as np
import time

size = 1_000_000
py_list = list(range(size))
np_array = np.arange(size)

start = time.time()
doubled = [x * 2 for x in py_list]
print(f"Python list: {time.time() - start:.4f} seconds")

start = time.time()
doubled = np_array * 2
print(f"NumPy array: {time.time() - start:.4f} seconds")
```

Run it on your own machine. The gap is usually between 20 and 100 times, and it
grows with the size of the data.

NumPy is fast because its operations run as compiled C over a block of memory
holding one data type, rather than as interpreted Python over a list of separate
objects. Everything in machine learning in Python sits on top of it: pandas,
scikit-learn, TensorFlow, PyTorch.

| | Python list | NumPy array |
|---|---|---|
| Speed | Interpreted, slow | Compiled, fast |
| Memory | An object per element | One compact block |
| Types | Mixed allowed | One type throughout |
| Operations | You write the loop | Applied to the whole array |

The habit to build is called **vectorisation**: stop thinking "loop over each
element" and start thinking "do this to the whole array".

## NumPy essentials

```python
import numpy as np
```

The alias `np` is universal. Use it.

### Creating arrays

```python
v = np.array([1, 2, 3, 4, 5])
M = np.array([[1, 2, 3],
              [4, 5, 6]])

print(v.shape, M.shape)
print(M.ndim, M.size, M.dtype)
```

```
(5,) (2, 3)
2 6 int64
```

Four attributes worth memorising now: `shape` is the dimensions, `ndim` is how
many there are, `size` is the total element count, `dtype` is the type they all
share.

Note `(5,)` with the trailing comma. That is a one dimensional array of five
elements, which is not the same object as a $5 \times 1$ matrix. The difference
causes real confusion later, so notice it now.

| Function | What you get |
|---|---|
| `np.zeros((3, 4))` | 3 by 4 of zeros |
| `np.ones((2, 3))` | 2 by 3 of ones |
| `np.eye(3)` | 3 by 3 identity, ones on the diagonal |
| `np.arange(0, 10, 2)` | Steps of 2, stopping before 10 |
| `np.linspace(0, 1, 5)` | Exactly 5 values, evenly spaced, endpoints included |
| `np.random.rand(3, 3)` | Uniform random between 0 and 1 |
| `np.random.randn(3, 3)` | Normal distribution, mean 0 |
| `np.random.randint(1, 100, 10)` | 10 random integers |

The difference between `arange` and `linspace` catches people out. With
`arange` you choose the step and NumPy decides how many values. With `linspace`
you choose how many and NumPy computes the step.

!!! tip "Seed your random numbers"

    ```python
    rng = np.random.default_rng(42)
    print(rng.random(3))
    ```

    Without a seed your results change on every run, which makes it impossible
    to tell whether a change you made helped or whether you got lucky. Seed
    anything you intend to compare.

### Indexing and slicing

```python
M = np.array([[5, 10, 15],
              [20, 25, 30],
              [35, 40, 45]])

print(M[1, 2])       # single element, row 1 column 2
print(M[1])          # a whole row
print(M[:, 1])       # a whole column
print(M[0:2, 1:3])   # the top right 2 by 2 block
```

```
30
[20 25 30]
[10 25 40]
[[10 15]
 [25 30]]
```

Use `M[row, col]`, not `M[row][col]`. It reads better and it is faster, because
the second form builds a temporary array for the row before indexing into it.

!!! warning "A slice is a view, not a copy"

    ```python
    original = np.array([1, 2, 3, 4, 5])
    piece = original[1:4]
    piece[0] = 999
    print(original)
    ```

    ```
    [  1 999   3   4   5]
    ```

    You changed `piece` and `original` changed with it. NumPy slices point into
    the same memory rather than duplicating it, which is what makes them fast.
    Python lists do not behave this way, so this will surprise you at least
    once.

    When you need an independent copy, say so: `piece = original[1:4].copy()`.

### Selecting by condition

```python
arr = np.arange(1, 11)
print(arr > 5)
print(arr[arr > 5])
print(arr[(arr > 3) & (arr < 8)])
```

```
[False False False False False  True  True  True  True  True]
[ 6  7  8  9 10]
[4 5 6 7]
```

The condition produces an array of True and False, and using it as an index
keeps the True positions. This is how you filter data for the rest of your
career, and it is one line instead of a loop.

Use `&` and `|` between conditions, not `and` and `or`, and keep each condition
in brackets. Python's `and` works on single true or false values and will raise
an error on an array.

### Arithmetic on whole arrays

```python
arr = np.arange(0, 5)
print(arr + 10)
print(arr * 2)
print(arr + arr)
print(arr * arr)
```

```
[10 11 12 13 14]
[0 2 4 6 8]
[0 2 4 6 8]
[ 0  1  4  9 16]
```

No loops anywhere. The same functions apply element by element across an entire
array: `np.sqrt`, `np.exp`, `np.log`, `np.sin`.

!!! warning "`*` is not matrix multiplication"

    ```python
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    print(A * B)    # element by element
    print(A @ B)    # matrix multiplication
    ```

    ```
    [[ 5 12]
     [21 32]]
    [[19 22]
     [43 50]]
    ```

    Both run without error, which is what makes this dangerous. If you wanted a
    matrix product and typed `*`, nothing warns you. You get plausible numbers
    that are wrong. Use `@` for matrix multiplication, or `np.dot`.

Compare the whole of your hand written work with its NumPy equivalent:

| By hand | NumPy |
|---|---|
| `add_vectors(v1, v2)` | `v1 + v2` |
| `scalar_multiply(2, v)` | `2 * v` |
| `dot_product(v1, v2)` | `v1 @ v2` |
| `transpose(M)` | `M.T` |
| `matrix_multiply(A, B)` | `A @ B` |

Twenty lines of loops replaced by one character each. That is the whole argument
for NumPy.

### Broadcasting

When arrays have different shapes, NumPy will stretch the smaller one to fit,
provided the shapes are compatible.

<svg viewBox="0 0 680 250" role="img" aria-labelledby="bc-title bc-desc" style="width:100%;height:auto;margin:1.2rem 0;font-family:var(--md-text-font-family, system-ui, sans-serif)">
<title id="bc-title">Broadcasting a row across a matrix</title>
<desc id="bc-desc">A three by three matrix plus a one dimensional array of three values. The smaller array is applied to every row of the matrix, giving a three by three result.</desc>
<rect x="30" y="66" width="34" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="64" y="66" width="34" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="98" y="66" width="34" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="30" y="96" width="34" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="64" y="96" width="34" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="98" y="96" width="34" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="30" y="126" width="34" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="64" y="126" width="34" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="98" y="126" width="34" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="47" y="86" text-anchor="middle" font-size="12" fill="var(--md-default-fg-color)">1</text>
<text x="81" y="86" text-anchor="middle" font-size="12" fill="var(--md-default-fg-color)">2</text>
<text x="115" y="86" text-anchor="middle" font-size="12" fill="var(--md-default-fg-color)">3</text>
<text x="47" y="116" text-anchor="middle" font-size="12" fill="var(--md-default-fg-color)">4</text>
<text x="81" y="116" text-anchor="middle" font-size="12" fill="var(--md-default-fg-color)">5</text>
<text x="115" y="116" text-anchor="middle" font-size="12" fill="var(--md-default-fg-color)">6</text>
<text x="47" y="146" text-anchor="middle" font-size="12" fill="var(--md-default-fg-color)">7</text>
<text x="81" y="146" text-anchor="middle" font-size="12" fill="var(--md-default-fg-color)">8</text>
<text x="115" y="146" text-anchor="middle" font-size="12" fill="var(--md-default-fg-color)">9</text>
<text x="81" y="178" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-graphite)">shape (3, 3)</text>
<text x="152" y="116" text-anchor="middle" font-size="18" fill="var(--h-graphite)">+</text>
<rect x="176" y="66" width="34" height="30" fill="none" stroke="var(--h-cherry)" stroke-dasharray="4 3" opacity="0.5"/>
<rect x="210" y="66" width="34" height="30" fill="none" stroke="var(--h-cherry)" stroke-dasharray="4 3" opacity="0.5"/>
<rect x="244" y="66" width="34" height="30" fill="none" stroke="var(--h-cherry)" stroke-dasharray="4 3" opacity="0.5"/>
<rect x="176" y="96" width="34" height="30" fill="var(--h-cherry)"/>
<rect x="210" y="96" width="34" height="30" fill="var(--h-cherry)"/>
<rect x="244" y="96" width="34" height="30" fill="var(--h-cherry)"/>
<rect x="176" y="126" width="34" height="30" fill="none" stroke="var(--h-cherry)" stroke-dasharray="4 3" opacity="0.5"/>
<rect x="210" y="126" width="34" height="30" fill="none" stroke="var(--h-cherry)" stroke-dasharray="4 3" opacity="0.5"/>
<rect x="244" y="126" width="34" height="30" fill="none" stroke="var(--h-cherry)" stroke-dasharray="4 3" opacity="0.5"/>
<text x="193" y="86" text-anchor="middle" font-size="12" fill="var(--h-cherry)" opacity="0.6">10</text>
<text x="227" y="86" text-anchor="middle" font-size="12" fill="var(--h-cherry)" opacity="0.6">20</text>
<text x="261" y="86" text-anchor="middle" font-size="12" fill="var(--h-cherry)" opacity="0.6">30</text>
<text x="193" y="116" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">10</text>
<text x="227" y="116" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">20</text>
<text x="261" y="116" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">30</text>
<text x="193" y="146" text-anchor="middle" font-size="12" fill="var(--h-cherry)" opacity="0.6">10</text>
<text x="227" y="146" text-anchor="middle" font-size="12" fill="var(--h-cherry)" opacity="0.6">20</text>
<text x="261" y="146" text-anchor="middle" font-size="12" fill="var(--h-cherry)" opacity="0.6">30</text>
<text x="227" y="178" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-cherry)">shape (3,)</text>
<text x="298" y="116" text-anchor="middle" font-size="18" fill="var(--h-graphite)">=</text>
<rect x="322" y="66" width="38" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="360" y="66" width="38" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="398" y="66" width="38" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="322" y="96" width="38" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="360" y="96" width="38" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="398" y="96" width="38" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="322" y="126" width="38" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="360" y="126" width="38" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<rect x="398" y="126" width="38" height="30" fill="var(--h-surface)" stroke="var(--h-surface-line)"/>
<text x="341" y="86" text-anchor="middle" font-size="12" fill="var(--md-default-fg-color)">11</text>
<text x="379" y="86" text-anchor="middle" font-size="12" fill="var(--md-default-fg-color)">22</text>
<text x="417" y="86" text-anchor="middle" font-size="12" fill="var(--md-default-fg-color)">33</text>
<text x="341" y="116" text-anchor="middle" font-size="12" fill="var(--md-default-fg-color)">14</text>
<text x="379" y="116" text-anchor="middle" font-size="12" fill="var(--md-default-fg-color)">25</text>
<text x="417" y="116" text-anchor="middle" font-size="12" fill="var(--md-default-fg-color)">36</text>
<text x="341" y="146" text-anchor="middle" font-size="12" fill="var(--md-default-fg-color)">17</text>
<text x="379" y="146" text-anchor="middle" font-size="12" fill="var(--md-default-fg-color)">28</text>
<text x="417" y="146" text-anchor="middle" font-size="12" fill="var(--md-default-fg-color)">39</text>
<text x="379" y="178" text-anchor="middle" font-size="12" font-weight="700" fill="var(--h-graphite)">shape (3, 3)</text>
<text x="470" y="90" font-size="12.5" fill="var(--h-graphite)">The row of three is applied</text>
<text x="470" y="108" font-size="12.5" fill="var(--h-graphite)">to every row of the matrix.</text>
<text x="470" y="134" font-size="12.5" fill="var(--h-graphite)">Nothing is actually copied</text>
<text x="470" y="152" font-size="12.5" fill="var(--h-graphite)">in memory. NumPy just</text>
<text x="470" y="170" font-size="12.5" fill="var(--h-graphite)">reuses the same values.</text>
<text x="340" y="226" text-anchor="middle" font-size="12" fill="var(--h-graphite)">shapes are compared from the right: each pair must be equal, or one of them must be 1</text>
</svg>

```python
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
offsets = np.array([10, 20, 30])

print(M + offsets)
```

```
[[11 22 33]
 [14 25 36]
 [17 28 39]]
```

This is how you standardise a dataset in one line: subtract the mean of each
column, divide by its standard deviation, with no loop over rows.

### The axis argument

Most NumPy functions can work down columns or across rows, and you choose which
with `axis`.

```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])

print(M.sum())          # everything
print(M.sum(axis=0))    # down the columns
print(M.sum(axis=1))    # across the rows
```

```
21
[5 7 9]
[6 15]
```

The rule that makes this stick: `axis=0` moves down through the rows, so you get
one result per column. `axis=1` moves across the columns, so you get one result
per row.

Since rows are your samples and columns are your features, `axis=0` is almost
always the one you want when computing statistics about a feature.

### Reshaping

```python
arr = np.arange(12)
print(arr.reshape(3, 4))
print(arr.reshape(3, -1).shape)
```

```
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
(3, 4)
```

The total number of elements has to stay the same. The `-1` means "work this one
out for me", which saves arithmetic and mistakes.

## When it goes wrong

??? note "`ValueError: matmul: Input operand 1 has a mismatch...`"

    Your shapes do not line up. Print both shapes before the operation. Often a
    transpose fixes it: `A @ B.T`. If the numbers still do not make sense, one
    of your arrays is oriented the wrong way round and the transpose is hiding
    the real problem.

??? note "The numbers are wrong but there was no error"

    Check whether you used `*` where you meant `@`. This is the most expensive
    silent bug on this page.

??? note "I changed one array and another one changed too"

    You took a slice, which is a view into the same memory. Use `.copy()` when
    you want an independent array.

??? note "`cannot reshape array of size 30 into shape (4,8)`"

    Four times eight is 32, not 30. Count your elements with `arr.size`, and use
    `-1` for one of the dimensions to avoid doing this arithmetic yourself.

??? note "Adding two arrays gave me a much bigger array"

    You probably added shapes `(5,)` and `(5, 1)`. Broadcasting aligns from the
    right, so those become `(5, 5)`. Print both shapes. If one is a column when
    you meant a flat array, `ravel()` flattens it.

## Check yourself

1. Build a $3 \times 4$ NumPy array of your own numbers. Print its shape,
   `ndim`, `size` and `dtype`, and say in one sentence what each means.
2. Compute the dot product of `[1, 0]` and `[0, 1]` by hand, then with NumPy.
   Explain the result using the diagram above.
3. Take a $4 \times 3$ matrix and subtract the mean of each column from every
   row, in one line, using broadcasting and `axis`.
4. Multiply a $2 \times 3$ by a $3 \times 4$ with `@`. Then try `*` on the same
   pair and write down what happens and why.
5. Slice a row out of an array, change one value in the slice, and print the
   original. Then do it again with `.copy()` and explain the difference.

## Quick reference

| Task | NumPy |
|---|---|
| Create from a list | `np.array([1, 2, 3])` |
| Shape, dimensions, type | `a.shape`, `a.ndim`, `a.dtype` |
| Zeros, ones, identity | `np.zeros((2,3))`, `np.ones(4)`, `np.eye(3)` |
| Ranges | `np.arange(0, 10, 2)`, `np.linspace(0, 1, 5)` |
| Random, reproducible | `rng = np.random.default_rng(42)` |
| Element, row, column | `M[1, 2]`, `M[1]`, `M[:, 1]` |
| Filter by condition | `a[a > 5]` |
| Element by element | `a + b`, `a * b`, `np.sqrt(a)` |
| Matrix multiplication | `A @ B` |
| Transpose | `A.T` |
| Per column, per row | `M.sum(axis=0)`, `M.sum(axis=1)` |
| Change shape | `a.reshape(3, -1)` |
| Independent copy | `a.copy()` |

## Questions to sit with

1. The dot product of two vectors is zero when they are at right angles. If two
   features of a dataset have a dot product near zero, what might that tell you
   about them?
2. NumPy arrays hold one type and Python lists hold anything. What does NumPy
   gain from that restriction, and what does it cost you?
3. A slice is a view rather than a copy. That choice makes NumPy faster and also
   makes a whole class of bugs possible. Would you have made the same trade?
4. Broadcasting silently makes two differently shaped arrays work together.
   Name a case where you would rather it raised an error instead.

## Next

You can move numbers. Next you put names on them, which is what turns an array
into a dataset.

[Pandas](pandas.md){ .h-button }

Worth your time outside this page: the
[3Blue1Brown linear algebra series](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
is the best visual explanation of these ideas that exists, and the
[NumPy beginner guide](https://numpy.org/doc/stable/user/absolute_beginners.html)
is the official reference.
