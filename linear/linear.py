

#
# class vector
# A Python object for an n-dimensional vector
#

from math import sin, cos, asin, acos

class vector:
    """A Python object for an n-dimensional vector."""
    def __init__(self, *components):
        if len(components)==0:
            self.components = (0,0,0)
        elif len(components) == 1:
            raise ValueError("Vectors need more than one component")
        else:
            self.components = components

        self.N = len(self.components)
        self.x = self.components[0]
        self.y = self.components[1]
        if self.N >= 3: self.z = self.components[2]

    def __repr__(self):
        if self.N == 2:
            return "<{}, {}>".format(*self.components)
        else:
            first = "<{}".format(self.components[0])
            mid_list = [", {},".format(e) for e in self.components[1:len(self.components)-1]]
            end = " {}>".format(self.components[-1])

            outStr = first
            for e in mid_list:
                outStr = outStr + e
            outStr = outStr + end
            return outStr

    def __str__(self):
        return self.__repr__()

    def mag(self):
        """Return the magnitude of a vector."""
        val = 0
        for i in self.components:
            val += i**2
        return val**(0.5)

    def unit(self):
        """Return a normalized vector of length one."""
        return vector(*[x/self.mag() for x in self.components])

    def __mul__(self, other):
        return vector(*[other*i for i in self.components])

    def __rmul__(self, other):
        return self*other

    def __add__(self, other):
        new = []
        for i, e in enumerate(self.components):
            new.append(e + other.components[i])
        return vector(*new)

    def __radd__(self, other):
        return self+other

    def __sub__(self, other):
        new = []
        for i,e in enumerate(self.components):
            new.append(e - other.components[i])
        return vector(*new)

    def __rsub__(self, other):
        return self-other

    def change_element(self, index, value):
        """Change the component of a vector at a specified index."""
        new = []
        for i,e in enumerate(self.components):
            if i == index:
                new.append(value)
            else:
                new.append(e)
        return vector(*new) # self = vector(*new) ; return self?

    def to_array(self, tup=False):
        """Output the vector's components as a list. Enter kwarg tup=True to output as tuple."""
        if tup:
            return tuple(self.components)
        else:
            return self.components

#
#
# General functions for vector objects
#
#

def zero_vec(dim):
    """Return a zero vector of a specified dimension."""
    new = []
    for i in range(0, dim): new.append(0)
    return vector(*new)

def mag(x):
    """Return the magnitude of a vector."""
    val = 0
    for i in x.components:
        val += i**2
    return val**(0.5)

def unit(x):
    """Return a vector of the same direction with length one."""
    return vector(*[i/mag(x) for i in x.components])

def dot(vec1, vec2):
    """Compute the dot product of two vectors."""
    # Check that types are both vector
    if not (isinstance(vec1, vector) and isinstance(vec2, vector)):
        raise TypeError("Arguments must both be vectors")
    # Check that dimensions are consistent
    elif vec1.N != vec2.N:
        raise ValueError("Vectors have dimensions {} and {}; vectors must have same dimension".format(vec1.N, vec2.N))
    else:
        prod = 0
        for i, e in enumerate(vec1.components):
            prod += e * vec2.components[i]
        return prod

def cross(u, v):
    """Compute the cross product of two vectors."""
    # Check that types are both vector
    if not (isinstance(u, vector) and isinstance(v, vector)):
        raise TypeError("Arguments must both be vectors")
    # Check that dimensions are consistent
    elif u.N != 3 or v.N != 3:
        raise ValueError("Both vectors must be of dimension 3")
    else:
        new = [u.y*v.z - v.y*u.z, v.x*u.z - u.x*v.z, u.x*v.y - v.x*u.y]
        return vector(*new)

def plane(point, u, v):
    """Find the equation of a plane that is spanned by two vectors at a  specified point."""
    normal = cross(u,v)
    if normal == vector(0,0,0):
        raise ValueError("Vectors cannot be parallel")

    a, b, c = normal.components
    d = -a*point[0] - b*point[1] - c*point[2]
    print("Equation for the plane: {}x + {}y + {}z - ({}) = 0".format(a,b,c,d))

def angle(u, v):
    """Return the geometric angle, in radians, between two vectors."""
    # Check that types are both vector
    if not (isinstance(u, vector) and isinstance(v, vector)):
        raise TypeError("Arguments must both be vectors")
    elif u.N != v.N:
        raise ValueError("Vectors have inconsistent dimension")

    costheta = dot(u,v)/(u.mag() * v.mag())
    return acos(costheta)

def is_ortho(u, v):
    """Check if two vectors are orthogonal using a dot product."""
    if dot(u,v) == 0:
        return True
    else:
        return False


#
# class matrix
# A Python matrix object
#

class matrix:
    """A Python matrix object."""
    def __init__(self, *frame):
        # m x n matrix
        self.frame = list(frame)
        self.nrow = len(frame)
        self.ncol = len(frame[0])

        # Check if rows have same length
        for row in frame:
            if len(row) != self.ncol:
                raise ValueError("Inconsistent row length on row {}".format(row))

    def __repr__(self):
        outString = ""
        for row_index, row in enumerate(self.frame):
            rowString = "[".ljust(1, " ")
            rowString += row_str(row)
            rowString += "]\n".rjust(1, " ")
            outString += rowString
        return outString

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        if isinstance(other, matrix):
            # Check if matrices are same size
            if self.nrow == other.nrow and self.ncol == other.ncol:
                new = []
                for row_index, row in enumerate(self.frame):
                    new_row = []
                    for i, entry in enumerate(row):
                        new_row.append(entry + other.frame[row_index][i])
                    new.append(new_row)
                return matrix(*new)
        else:
            raise TypeError("You cannot add type {} to a matrix".format(type(other)))

    def __radd__(self, other):
        return self+other

    def __sub__(self, other):
        if isinstance(other, matrix):
            # Check if matrices are same size
            if self.nrow == other.nrow and self.ncol == other.ncol:
                new = []
                for row_index, row in enumerate(self.frame):
                    new_row = []
                    for i, entry in enumerate(row):
                        new_row.append(entry - other.frame[row_index][i])
                    new.append(new_row)
                return matrix(*new)

    def T(self):
        """Return the transpose of the object matrix."""
        return matrix(*[[row[i] for row in self.frame] for i in range(len(self.frame[0]))])

    def __rsub__(self, other):
        return self-other

    def __mul__(self, other):
        if type(other) is int or type(other) is float:
            new = []
            for row in self.frame:
                new_row = []
                for entry in row:
                    new_row.append(other*entry)
                new.append(new_row)
            return matrix(*new)

        elif isinstance(other, matrix) or isinstance(other, vector):
            # Check types
            if isinstance(other, vector):
                other = vector_to_matrix(other)
            # check if dimensions are consistent
            if self.ncol != other.nrow:
                raise ValueError("Dimensions are inconsistent; cannot multiply matrices")
            # Do the matrix multiplication
            new = []
            other = other.T()
            for row in self.frame:
                new_row = []
                for oth_row in other.frame:
                    new_row.append(row_mult(row, oth_row))
                new.append(new_row)
            return matrix(*new)


    def __rmul__(self, other):
        return self*other

    def is_square(self):
        """Check if the matrix is square."""
        if self.nrow == self.ncol:
            return True
        else:
            return False

    def rref(self):
        """Return the row-reduced Echelon form of a matrix."""
        M = self.frame
        lead = 0
        for r in range(self.nrow):
            if lead >= self.ncol:
                return matrix(*M)
            i = r
            while M[i][lead] == 0:
                i += 1
                lead += 1
                if self.ncol == lead:
                    return matrix(*M)
            M[i], M[r] = M[r], M[i]
            lv = M[r][lead]
            M[r] = [mrx/float(lv) for mrx in M[r]]
            for i in range(self.nrow):
                if i != r:
                    lv = M[i][lead]
                    M[i] = [iv - lv*rv for rv, iv in zip(M[r], M[i])]
            lead += 1
        return matrix(*M)


    def det(self):
        """Compute the matrix's determinant. Algorithm adopted from Sachin Joglekar, ActiveState Code Recipes."""
        # Check that matrix is square:
        if self.is_square() == False:
            raise ValueError("Cannot compute determinant of non-square matrix")
        # Compute det
        n = self.nrow
        if (n > 2):
            i = 1
            t = 0
            sum = 0
            while t <= n-1:
                d = {}
                t1 = 1
                while t1 <= n-1:
                    m = 0
                    d[t1] = []
                    while m <= n-1:
                        if (m==t):
                            u = 0
                        else:
                            d[t1].append(self.frame[t1][m])
                        m += 1
                    t1 += 1
                l1 = [d[x] for x in d]
                sum = sum + i*(self.frame[0][t])*(matrix(*l1).det()) ### l1 is a list
                i = i*(-1)
                t += 1
            return sum
        else:
            return (self.frame[0][0]*self.frame[1][1] - self.frame[0][1]*self.frame[1][0])

    def inv(self):
        """Invert the matrix."""
        # Check if matrix is square
        if self.is_square() == False:
            raise ValueError("Cannot invert non-square matrix")

        # Augment self with identity matrix
        new_mat = []
        for row_index, row in enumerate(self.frame):
            new_mat.append(row + identity_row(row_index, self.ncol))

        # Row reduce
        interim = matrix(*new_mat)
        interim = interim.rref()

        # Extract right half
        return_mat = []
        for i, row in enumerate(interim.frame):
            new_row = []
            for j, val in enumerate(row):
                if j >= interim.ncol/2:
                    new_row.append(val)
            return_mat.append(new_row)

        return matrix(*return_mat)

    def imt(mat):
        if mat.det() != 0:
            return True
        else:
            return False

def row_str(nums):
    outstr = ""
    for i in nums:
        outstr = outstr + "  {:6.3f}  ".format(i)
    return outstr

def identity_row(index, length):
    row = []
    for i in range(0, length):
        if i == index:
            row.append(1)
        else:
            row.append(0)
    return row

def row_mult(row1, row2):
    """Compute the inner product of two Python iterables."""
    entry = 0
    for i,e in enumerate(row1):
        entry += e * row2[i]
    return entry

def vector_to_matrix(u):
    """Convert a vector object into a single-column matrix."""
    new = []
    for comp in u.components: new.append([comp])
    return matrix(*new)

def id(N):
    """Return a square identity matrix with N number of rows/columns."""
    frame = []
    for i in range(0, N):
        row = []
        for j in range(0, N):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        frame.append(row)
    return matrix(*frame)

def zero(m, n):
    """Return a zero matrix with row number m and column number n."""
    frame = []
    for i in range(0, m):
        row = []
        for j in range(0, n):
            row.append(0)
        frame.append(row)
    return matrix(*frame)

def commute(matrix1, matrix2):
    """Check if two matrices commute under multiplication."""
    if (matrix1*matrix2).frame == (matrix2*matrix1).frame:
        return True
    else:
        return False

def least_squares(A, b):
    """Compute the least squares solution to Ax = b."""
    xhat = (A.T() * A).inv() * A.T() * b
    error = b - A*xhat
    return xhat, error

def in_null(A, x):
    """Check if a vector x is in the null space of A."""
    test = A*x
    if test.frame == zero(test.nrow, test.ncol).frame:
        return True
    else:
        return False

def in_col(A, x):
    """Check if a vector x in the column space of A."""
    test = A*x
    if test.frame == zero(test.nrow, test.ncol).frame:
        return False
    else:
        return True
