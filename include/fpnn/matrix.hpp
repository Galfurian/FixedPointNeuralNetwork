#pragma once

#include <vector>
#include <ostream>
#include <cassert>

namespace fpnn
{
template <typename T>
class Matrix {
private:
    std::vector<T> _data;
    unsigned _rows;
    unsigned _cols;

    typedef std::initializer_list<std::initializer_list<T>> initializer_list_type_t;

public:
    /// @brief Construct a new Matrix object.
    /// @param rows
    /// @param cols
    /// @param initial
    Matrix(unsigned rows, unsigned cols, const T &initial);

    /// @brief Construct a new Matrix object.
    /// @param values
    Matrix(const initializer_list_type_t &values);

    /// @brief Construct a new Matrix object.
    /// @param rhs the other matrix.
    Matrix(const Matrix<T> &rhs);

    /// @brief Destroy the Matrix object.
    virtual ~Matrix();

    /// @brief Get the number of rows of the matrix.
    /// @return the number of rows.
    inline unsigned get_rows() const
    {
        return _rows;
    }

    /// @brief Get the number of columns of the matrix.
    /// @return the number of columns.
    inline unsigned get_cols() const
    {
        return _cols;
    }

    /// @brief Returns the total size of the matrix.
    /// @return the total size of the matrix.
    inline unsigned get_size() const
    {
        return _rows * _cols;
    }

    /// @brief Assign operator.
    /// @param rhs the other matrix.
    /// @return a reference to this matrix.
    Matrix<T> &operator=(const Matrix<T> &rhs);

    /// @brief Addition of two matrices.
    /// @param rhs the other matrix.
    /// @return the result of the operation.
    Matrix<T> operator+(const Matrix<T> &rhs);

    /// @brief Cumulative addition of this matrix and another.
    /// @param rhs the other matrix.
    /// @return a reference to this matrix.
    Matrix<T> &operator+=(const Matrix<T> &rhs);

    /// @brief Subtraction of this matrix and another.
    /// @param rhs the other matrix.
    /// @return the result of the operation.
    Matrix<T> operator-(const Matrix<T> &rhs);

    /// @brief Cumulative subtraction of this matrix and another.
    /// @param rhs the other matrix.
    /// @return a reference to this matrix.
    Matrix<T> &operator-=(const Matrix<T> &rhs);

    /// @brief Left multiplication of this matrix and another.
    /// @param rhs the other matrix.
    /// @return the result of the operation.
    Matrix<T> operator*(const Matrix<T> &rhs);

    /// @brief Cumulative left multiplication of this matrix and another.
    /// @param rhs the other matrix.
    /// @return a reference to this matrix.
    Matrix<T> &operator*=(const Matrix<T> &rhs);

    /// @brief Right division of this matrix and another.
    /// @param rhs the other matrix.
    /// @return the result of the operation.
    Matrix<T> operator/(const Matrix<T> &rhs);

    /// @brief Cumulative right division of this matrix and another.
    /// @param rhs the other matrix.
    /// @return a reference to this matrix.
    Matrix<T> &operator/=(const Matrix<T> &rhs);

    /// @brief Matrix/scalar addition.
    /// @param rhs the other matrix.
    /// @return the result of the operation.
    Matrix<T> operator+(const T &rhs);

    /// @brief Cumulative matrix/scalar addition.
    /// @param rhs the other matrix.
    /// @return the result of the operation.
    Matrix<T> &operator+=(const T &rhs);

    /// @brief Matrix/scalar subtraction.
    /// @param rhs the other matrix.
    /// @return the result of the operation.
    Matrix<T> operator-(const T &rhs);

    /// @brief Cumulative matrix/scalar subtraction.
    /// @param rhs the other matrix.
    /// @return the result of the operation.
    Matrix<T> &operator-=(const T &rhs);

    /// @brief Matrix/scalar multiplication.
    /// @param rhs the other matrix.
    /// @return the result of the operation.
    Matrix<T> operator*(const T &rhs);

    /// @brief Cumulative matrix/scalar multiplication.
    /// @param rhs the other matrix.
    /// @return the result of the operation.
    Matrix<T> &operator*=(const T &rhs);

    /// @brief Matrix/scalar division.
    /// @param rhs the other matrix.
    /// @return the result of the operation.
    Matrix<T> operator/(const T &rhs);

    /// @brief Cumulative matrix/scalar division.
    /// @param rhs the other matrix.
    /// @return the result of the operation.
    Matrix<T> &operator/=(const T &rhs);

    /// @brief Multiply a matrix with a vector.
    /// @param rhs the other matrix.
    /// @return
    std::vector<T> operator*(const std::vector<T> &rhs);

    /// @brief Calculate a transpose of this matrix.
    /// @return the transposed matrix.
    Matrix<T> transpose() const;

    /// @brief Calculates the determinant of this matrix.
    /// @return the determinant.
    T determinant() const;

    /// @brief Calculates the adjoint of this matrix.
    /// @return the adjoint.
    Matrix<T> adjoint() const;

    /// @brief Calculates the inverse of this matrix.
    /// @return the inverse.
    Matrix<T> inverse() const;

    /// @brief Extracts the diagonal elements.
    /// @return the diagonal elements.
    std::vector<T> diag() const;

    /// @brief Operator for accessing the matrix.
    /// @param row the accessed row.
    /// @param col the accessed column.
    /// @return the reference to the accessed item.
    T &operator()(const unsigned &row, const unsigned &col);

    /// @brief Operator for accessing the matrix.
    /// @param row the accessed row.
    /// @param col the accessed column.
    /// @return the const reference to the accessed item.
    const T &operator()(const unsigned &row, const unsigned &col) const;

    /// @brief Alternative function to access the matrix.
    /// @param row the accessed row.
    /// @param col the accessed column.
    /// @return the reference to the accessed item.
    inline T &at(const unsigned &row, const unsigned &col)
    {
        return this->operator()(row, col);
    }

    /// @brief Alternative function to access the matrix.
    /// @param row the accessed row.
    /// @param col the accessed column.
    /// @return the const reference to the accessed item.
    inline const T &at(const unsigned &row, const unsigned &col) const
    {
        return this->operator()(row, col);
    }
};

template <typename T>
Matrix<T>::Matrix(unsigned rows, unsigned cols, const T &initial)
    : _data(rows * cols, initial),
      _rows(rows),
      _cols(cols)
{
    // Nothing to do.
}

template <typename T>
Matrix<T>::Matrix(const initializer_list_type_t &values)
    : _data(),
      _rows(),
      _cols()
{
    // Get the rows.
    auto rows = values.size();
    // Get the list of rows.
    const auto &row_list = values.begin();
    if (rows > 0) {
        // Get the cols.
        auto cols = values.begin()->size();
        if (cols > 0) {
            // Set rows and cols.
            _rows = rows, _cols = cols;
            // Resize the matrix.
            _data.resize(_rows * _cols);

            for (size_t row_index = 0; row_index < _rows; ++row_index) {
                // Set the cols.
                cols = row_list[row_index].size();
                if (_cols != cols) {
                    std::cerr << "Row " << row_index << " has wrong number of elements.\n";
                    _rows = _cols = 0;
                    _data.clear();
                    return;
                }
                // Get the list of cells in the row.
                const auto &column_list = row_list[row_index].begin();
                for (size_t column_index = 0; column_index < _cols; ++column_index) {
                    this->at(row_index, column_index) = column_list[column_index];
                }
            }
        }
    }
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T> &rhs)
    : _data(rhs._data),
      _rows(rhs.get_rows()),
      _cols(rhs.get_cols())
{
    // Nothing to do.
}

template <typename T>
Matrix<T>::~Matrix()
{
}

template <typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &rhs)
{
    if (&rhs == this)
        return *this;
    // Set the new rows and columns.
    _rows = rhs.get_rows();
    _cols = rhs.get_cols();
    // Resize the matrix.
    _data.resize(_rows * _cols);
    for (unsigned i = 0; i < _rows; i++)
        for (unsigned j = 0; j < _cols; j++)
            this->at(i, j) = rhs(i, j);
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &rhs)
{
    unsigned rows = rhs.get_rows(), cols = rhs.get_cols();
    bool lhs_is_cell = ((_rows == 1) && (_cols == 1)),
         rhs_is_cell = ((rows == 1) && (cols == 1));
    bool lhs_is_vect = ((_rows == 1) || (_cols == 1)) && !lhs_is_cell,
         rhs_is_vect = ((rows == 1) || (cols == 1)) && !rhs_is_cell;
    if (lhs_is_cell || rhs_is_cell) {
        // Prepare the result.
        Matrix result(rhs_is_cell ? _rows : rows, rhs_is_cell ? _cols : cols, 0.0);
        // Prepare the references to the right matrices.
        const Matrix<T> &_matrix = rhs_is_cell ? *this : rhs, _cell = rhs_is_cell ? rhs : *this;
        // Compute the sum.
        for (size_t i = 0; i < (rhs_is_cell ? this->get_size() : rhs.get_size()); ++i)
            result._data[i] = _matrix._data[i] + _cell._data[0];
        return result;
    } else if (lhs_is_vect || rhs_is_vect) {
        // Prepare the result.
        Matrix result(rhs_is_vect ? _rows : rows, rhs_is_vect ? _cols : cols, 0.0);
        // Prepare the references to the right matrices.
        const Matrix<T> &_matrix = rhs_is_vect ? *this : rhs, _vect = rhs_is_vect ? rhs : *this;
        // Compute the selector.
        bool is_row_vect = (_vect._rows == 1);
        // Compute the sum.
        for (unsigned i = 0; i < (rhs_is_vect ? _rows : rows); i++)
            for (unsigned j = 0; j < (rhs_is_vect ? _cols : cols); j++)
                result(i, j) = _matrix(i, j) + _vect(i * !is_row_vect, j * is_row_vect);
        return result;
    } else if ((_rows == rows) && (_cols == cols)) {
        Matrix result(_rows, _cols, 0.0);
        for (unsigned i = 0; i < _rows; i++)
            for (unsigned j = 0; j < _cols; j++)
                result(i, j) = this->at(i, j) + rhs(i, j);
        return result;
    }
    assert(false && "Arrays have incompatible sizes for this operation.");
}

template <typename T>
Matrix<T> &Matrix<T>::operator+=(const Matrix<T> &rhs)
{
    return ((*this) = this->operator+(rhs));
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &rhs)
{
    unsigned rows = rhs.get_rows(), cols = rhs.get_cols();
    bool lhs_is_cell = ((_rows == 1) && (_cols == 1)),
         rhs_is_cell = ((rows == 1) && (cols == 1));
    bool lhs_is_vect = ((_rows == 1) || (_cols == 1)) && !lhs_is_cell,
         rhs_is_vect = ((rows == 1) || (cols == 1)) && !rhs_is_cell;
    if (lhs_is_cell || rhs_is_cell) {
        // Prepare the result.
        Matrix result(rhs_is_cell ? _rows : rows, rhs_is_cell ? _cols : cols, 0.0);
        // Prepare the references to the right matrices.
        const Matrix<T> &_matrix = rhs_is_cell ? *this : rhs, _cell = rhs_is_cell ? rhs : *this;
        // Compute the sum.
        for (size_t i = 0; i < (rhs_is_cell ? this->get_size() : rhs.get_size()); ++i)
            result._data[i] = _matrix._data[i] - _cell._data[0];
        return result;
    } else if (lhs_is_vect || rhs_is_vect) {
        // Prepare the result.
        Matrix result(rhs_is_vect ? _rows : rows, rhs_is_vect ? _cols : cols, 0.0);
        // Prepare the references to the right matrices.
        const Matrix<T> &_matrix = rhs_is_vect ? *this : rhs, _vect = rhs_is_vect ? rhs : *this;
        // Compute the selector.
        bool is_row_vect = (_vect._rows == 1);
        // Compute the sum.
        for (unsigned i = 0; i < (rhs_is_vect ? _rows : rows); i++)
            for (unsigned j = 0; j < (rhs_is_vect ? _cols : cols); j++)
                result(i, j) = _matrix(i, j) - _vect(i * !is_row_vect, j * is_row_vect);
        return result;
    } else if ((_rows == rows) && (_cols == cols)) {
        Matrix result(_rows, _cols, 0.0);
        for (unsigned i = 0; i < _rows; i++)
            for (unsigned j = 0; j < _cols; j++)
                result(i, j) = this->at(i, j) - rhs(i, j);
        return result;
    }
    assert(false && "Arrays have incompatible sizes for this operation.");
}

template <typename T>
Matrix<T> &Matrix<T>::operator-=(const Matrix<T> &rhs)
{
    return ((*this) = this->operator-(rhs));
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &rhs)
{
    unsigned rows = rhs.get_rows(), cols = rhs.get_cols();
    assert(_cols == rows && "For matrix multiplication, the number of columns in the first"
                            "matrix must be equal to the number of rows in the second matrix.");
    Matrix result(_rows, cols, 0.0);
    for (unsigned i = 0; i < _rows; i++) {
        for (unsigned j = 0; j < cols; j++) {
            for (unsigned k = 0; k < _rows; k++) {
                result(i, j) += this->at(i, k) * rhs(k, j);
            }
        }
    }
    return result;
}

template <typename T>
Matrix<T> &Matrix<T>::operator*=(const Matrix<T> &rhs)
{
    return ((*this) = this->operator*(rhs));
}

template <typename T>
Matrix<T> Matrix<T>::operator/(const Matrix<T> &rhs)
{
    return this->operator*(rhs.inverse());
}

template <typename T>
Matrix<T> &Matrix<T>::operator/=(const Matrix<T> &rhs)
{
    return ((*this) = this->operator*(rhs.inverse()));
}

template <typename T>
Matrix<T> Matrix<T>::transpose() const
{
    Matrix result(_cols, _rows, 0.0);
    for (unsigned i = 0; i < _rows; i++)
        for (unsigned j = 0; j < _cols; j++)
            result(j, i) = this->at(i, j);
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const T &rhs)
{
    Matrix result(_rows, _cols, 0.0);
    for (unsigned i = 0; i < _rows; i++)
        for (unsigned j = 0; j < _cols; j++)
            result(i, j) = this->at(i, j) + rhs;
    return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(const T &rhs)
{
    for (unsigned i = 0; i < _rows; i++)
        for (unsigned j = 0; j < _cols; j++)
            this->at(i, j) += rhs;
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const T &rhs)
{
    Matrix result(_rows, _cols, 0.0);
    for (unsigned i = 0; i < _rows; i++)
        for (unsigned j = 0; j < _cols; j++)
            result(i, j) = this->at(i, j) - rhs;
    return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=(const T &rhs)
{
    for (unsigned i = 0; i < _rows; i++)
        for (unsigned j = 0; j < _cols; j++)
            this->at(i, j) -= rhs;
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const T &rhs)
{
    Matrix result(_rows, _cols, 0.0);
    for (unsigned i = 0; i < _rows; i++)
        for (unsigned j = 0; j < _cols; j++)
            result(i, j) = this->at(i, j) * rhs;
    return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*=(const T &rhs)
{
    for (unsigned i = 0; i < _rows; i++)
        for (unsigned j = 0; j < _cols; j++)
            this->at(i, j) *= rhs;
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator/(const T &rhs)
{
    Matrix result(_rows, _cols, 0.0);
    for (unsigned i = 0; i < _rows; i++)
        for (unsigned j = 0; j < _cols; j++)
            result(i, j) = this->at(i, j) / rhs;
    return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator/=(const T &rhs)
{
    for (unsigned i = 0; i < _rows; i++)
        for (unsigned j = 0; j < _cols; j++)
            this->at(i, j) /= rhs;
    return *this;
}

template <typename T>
std::vector<T> Matrix<T>::operator*(const std::vector<T> &rhs)
{
    std::vector<T> result(rhs.size(), 0.0);
    for (unsigned i = 0; i < _rows; i++)
        for (unsigned j = 0; j < _cols; j++)
            result[i] = this->at(i, j) * rhs[j];
    return result;
}

template <typename T>
std::vector<T> Matrix<T>::diag() const
{
    std::vector<T> result(_rows, 0.0);
    for (unsigned i = 0; i < _rows; i++)
        result[i] = this->at(i, i);
    return result;
}

template <typename T>
T &Matrix<T>::operator()(const unsigned &row, const unsigned &col)
{
    return _data[(row * _cols) + col];
}

template <typename T>
const T &Matrix<T>::operator()(const unsigned &row, const unsigned &col) const
{
    return _data[(row * _cols) + col];
}

/// @brief Function to get cofactor of an input matrix.
/// @param input  The input matrix.
/// @param output The output matrix.
/// @param rows   The number of rows of the matrix to consider.
/// @param cols   The number of columns of the matrix to consider.
/// @param p      The row that must be removed.
/// @param q      The column that must be removed.
/// @return the input matrix without row p and column q.
template <typename T>
static inline Matrix<T> &__compute_minor(
    const Matrix<T> &input,
    Matrix<T> &output,
    size_t rows,
    size_t cols,
    size_t p,
    size_t q)
{
    // Looping for each element of the matrix.
    for (size_t i = 0, j = 0, row = 0, col = 0; row < rows; ++row) {
        for (col = 0; col < cols; ++col) {
            // Copying only those element which are not in given row and column.
            if ((row != p) && (col != q)) {
                output(i, j++) = input(row, col);
                // When the row is filled, increase row index and reset col index.
                if (j == (cols - 1))
                    j = 0, ++i;
            }
        }
    }
    return output;
}

/// @brief Computes the determinant, through recursion.
/// @param matrix  The input matrix.
/// @param N       The size of the matrix that must be considered.
/// @return the determinant of the original input matrix.
template <typename T>
static inline T __determinant_rec(const Matrix<T> &matrix, size_t N)
{
    // The matrix contains single element.
    if (N == 1)
        return matrix(0, 0);
    // The matrix contains only two elements.
    if (N == 2)
        return (matrix(0, 0) * matrix(1, 1)) - (matrix(0, 1) * matrix(1, 0));
    // The matrix contains more than two elements.
    int sign      = 1;
    T determinant = 0;
    Matrix<T> support(N, N, 0);
    for (size_t i = 0; i < N; ++i) {
        // Compute the minor of matrix[0][i]
        __compute_minor(matrix, support, N, N, 0, i);
        // Recursively call this function.
        determinant += sign * matrix(0, i) * __determinant_rec(support, N - 1);
        // Alternate the sign.
        sign = -sign;
    }
    return determinant;
}

template <typename T>
T Matrix<T>::determinant() const
{
    assert(this->get_cols() == this->get_rows() && "Matrix must be square.");
    return __determinant_rec(*this, this->get_cols());
}

template <typename T>
Matrix<T> Matrix<T>::adjoint() const
{
    assert(this->get_cols() == this->get_rows() && "Matrix must be square.");
    size_t N = this->get_cols();
    // Return 1.
    if (N == 1)
        return Matrix<T>(1, 1, 1);

    // temp is used to store cofactors of A[][]
    int sign = 1;
    Matrix<T> adj(N, N, 0), support(N, N, 0);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            // Get cofactor of A[i][j]
            __compute_minor(*this, support, N, N, i, j);

            // sign of adj[j][i] positive if sum of row
            // and column indexes is even.
            sign = ((i + j) % 2 == 0) ? 1 : -1;

            // Interchanging rows and columns to get the
            // transpose of the cofactor matrix
            adj(j, i) = sign * __determinant_rec(support, N - 1);
        }
    }
    return adj;
}

template <typename T>
Matrix<T> Matrix<T>::inverse() const
{
    assert(this->get_cols() == this->get_rows() && "Matrix must be square.");
    // Get the size of the matrix.
    size_t N = this->get_cols();
    // Compute the determinant.
    T det = this->determinant();
    // If determinant is zero, the matrix is singular.
    if (det == 0) {
        std::cerr << "Singular matrix, can't find its inverse.\n";
        return Matrix<T>(0, 0, 0);
    }
    // Find adjoint of the matrix.
    auto adjoint = this->adjoint();
    // Create a matrix for the result.
    Matrix<T> inverse(N, N, 0);
    // Find Inverse using formula "inverse(A) = adj(A)/det(A)".
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            inverse(i, j) = adjoint(i, j) / det;
    return inverse;
}

template <typename T>
std::ostream &operator<<(std::ostream &lhs, const Matrix<T> &rhs)
{
    for (unsigned row = 0; row < rhs.get_rows(); ++row) {
        for (unsigned col = 0; col < rhs.get_cols(); ++col) {
            lhs << rhs(row, col) << " ";
        }
        lhs << "\n";
    }
    return lhs;
}

} // namespace fpnn