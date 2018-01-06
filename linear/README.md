# linear - a python vector/matrix package
This python package includes classes and functions for working with vectors and matrices.

## Installation
To install with pip, simply do
```
pip install linear
```
To install with git, run the following:
```
git clone \dir
python setup.py install
```

## Using linear
There are two classes in linear: the vector class and the matrix class.

### Vectors
Declare a vector simply by arguing its components. For example:
```
vector(3,5,4,7,5)
>> <3, 5, 4, 7, 5>
```
The vector class has several attributes designed for ease of use:
- `vector.components`: python list containing the vector's components.
- `vector.x`, `vector.y`, `vector.z`: the first three components of a vector.
- `vector.N`: number of components in the vector.

The class also contains a variety of methods for working with vectors. Addition,
subtraction, and scalar multiplication are all overloaded.
- `vector.mag()`: Return the magnitude of the vector.
- `vector.unit()`: Return a vector of the same direction with length 1.
- `vector.change_element(index, value)`: Modify the vector to have `value` at location `index` in `vector.components`.
- `vector.to_array()`: Return the vector's components as a list (default) or tuple (specify `tup=True`).

The package includes the following functions for vector operations:
- `zero_vec(N)`: return a zero vector with `N` components.
- `mag(u)`: Return the magnitude of a vector `u`.
- `unit(u)`: Return the unit vector of `u`.
- `dot(u, v)`: Return the dot product of vectors `u` and `v`.
- `cross(u, v)`: Return the vector cross product of `u` and `v`.
- `is_ortho(u, v)`: Check if two vectors `u` and `v` are orthogonal. Returns True/False.
- `angle(u, v)`: Return the geometric angle, in radians, between the vectors `u` and `v`.
- `plane(point, u, v)`: Print the equation for the plane that is spanned by the two vectors
   at the specified point. `u` and `v` are instances of the vector class, and `point` is an iterable.

### Matrices
Declare matrices by entering the rows as individual lists, e.g.
```
matrix([3,5,1],[2,-1,-1],[3,5,0])
>>  [   3.000     5.000     1.000  ]
    [   2.000    -1.000    -1.000  ]
    [   3.000     5.000     0.000  ]

```
The matrix object has the following attributes:
 - `matrix.frame`: a two-dimensional python list containing the matrix rows
 - `matrix.nrow`: number of rows in the matrix
 - `matrix.ncol`: number of columns in the matrix

The matrix object itself includes many methods for doing linear algebra. These include:
 - `matrix.is_square()`: checks if a matrix is square. Returns True/False.
 - `matrix.T()`: return the transpose of the matrix.
 - `matrix.rref()`: Row-reduce the matrix into Echelon form. Algorithm adopted from Rosetta Code.
 - `matrix.det()`: Compute the determinant of the matrix using a recursive algorithm. Adopted from Sachin Joglekar.
 - `matrix.inv()`: Compute the inverse of the matrix.
 - `matrix.imt()`: Check if the matrix satisfies the invertible matrix theorem. Returns True/False.

Other functions for working with matrices:
 - `vector_to_matrix(u)`: convert a vector `u` into a single-column matrix.
 - `id(N)`: Return a square identity matrix with `N` number of rows and columns.
 - `zero(m, n)`: Return a zero matrix with row number `m` and column number `n`.
 - `commute(A, B)`: check if two matrices commute under multiplication. Returns True/False.
 - `least_squares(A, x)`: Find the least squares solution to Ax=b. Returns both the least-square solution and its error.
 - `in_null(A, x)`: Check if x is in the null space of A. Returns True/False.
 - `in_col(A, x)`: Check if x is in the column space of A. Returns True/False.
