#include <iostream>
#include "fpnn/fixed_point_matrix.hpp"
#include "fpnn/matrix_utility.hpp"
#include "fixedPoint.hpp"

template <typename MatrixType>
void _dump(const char *name, const MatrixType &m)
{
    std::cout << name << "[" << m.get_rows() << ", " << m.get_cols() << "] : \n"
              << m << "\n";
}
#define dump(M) _dump(#M, M)

template <typename MatrixType>
void test_op()
{
    {
        MatrixType m1{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}};
        MatrixType m2{{9}};

        auto sum = m1 + m2;
        auto sub = m1 - m2;

        dump(m1);
        dump(m2);
        dump(sum);
        dump(sub);
    }
    std::cout << std::string(40, '=') << "\n";
    {
        MatrixType m1{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}};
        MatrixType m2{
            {9, 6, 3}};

        auto sum = m1 + m2;
        auto sub = m1 - m2;

        dump(m1);
        dump(m2);
        dump(sum);
        dump(sub);
    }
    std::cout << std::string(40, '=') << "\n";
    {
        MatrixType m1{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}};
        MatrixType m2{
            {9},
            {6},
            {3}};

        auto sum = m1 + m2;
        auto sub = m1 - m2;
        auto mul = m1 * m2;

        dump(m1);
        dump(m2);
        dump(sum);
        dump(sub);
        dump(mul);
    }
    std::cout << std::string(40, '=') << "\n";
    {
        MatrixType m1{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}};
        MatrixType m2{
            {9, 8, 7},
            {6, 7, 4},
            {3, 2, 1}};

        auto sum = m1 + m2;
        auto sub = m1 - m2;
        auto mul = m1 * m2;

        dump(m1);
        dump(m2);
        dump(sum);
        dump(sub);
        dump(mul);
    }
    std::cout << std::string(40, '=') << "\n";
}

template <typename MatrixType>
void test_adv_op()
{
    {
        MatrixType m{
            {3, 2},
            {4, 3}};
        dump(m);
        std::cout << m.determinant() << " == " << 36 << "\n";
        dump(m.transpose());
        dump(m.adjoint());
        dump(m.inverse());
        dump(m * m.inverse());
    }
    std::cout << std::string(40, '=') << "\n";
    {
        MatrixType m{
            {2, 1, 3},
            {6, 5, 7},
            {4, 9, 8}};
        dump(m);
        std::cout << m.determinant() << " == " << 36 << "\n";
        dump(m.transpose());
        dump(m.adjoint());
        dump(m.inverse());
        dump(m * m.inverse());
    }
    std::cout << std::string(40, '=') << "\n";
    {
        MatrixType m{
            {4, 6},
            {3, 8}};
        dump(m);
        std::cout << m.determinant() << " == " << 14 << "\n";
        dump(m.transpose());
        dump(m.adjoint());
        dump(m.inverse());
        dump(m * m.inverse());
    }
    std::cout << std::string(40, '=') << "\n";
    {
        MatrixType m{
            {6, 1, 1},
            {4, -2, 5},
            {2, 8, 7}};
        dump(m);
        std::cout << m.determinant() << " == " << -306 << "\n";
        dump(m.transpose());
        dump(m.adjoint());
        dump(m.inverse());
        dump(m * m.inverse());
    }
    std::cout << std::string(40, '=') << "\n";
    {
        MatrixType m{
            {2, -3, 1},
            {2, 0, -1},
            {1, 4, 5}};
        dump(m);
        std::cout << m.determinant() << " == " << 49 << "\n";
        dump(m.transpose());
        dump(m.adjoint());
        dump(m.inverse());
        dump(m * m.inverse());
    }
    std::cout << std::string(40, '=') << "\n";
    {
        auto m = fpnn::eye<MatrixType>(4u) * 2;
        dump(m);
        std::cout << m.determinant() << " == " << 49 << "\n";
        dump(m.transpose());
        dump(m.adjoint());
        dump(m.inverse());
        dump(m * m.inverse());
    }
    std::cout << std::string(40, '=') << "\n";
    {
        MatrixType m{
            {5, -2, 2, 7},
            {1, 0, 0, 3},
            {-3, 1, 5, 0},
            {3, -1, -9, 4}};
        dump(m);
        std::cout << m.determinant() << " == " << 88 << "\n";
        dump(m.transpose());
        dump(m.adjoint());
        dump(m.inverse());
        dump(m * m.inverse());
    }
    std::cout << std::string(40, '=') << "\n";
}

int main(int argc, char *argv[])
{
    test_op<QSMatrix<FixedPoint<32, 32>>>();
    test_adv_op<QSMatrix<FixedPoint<32, 32>>>();
    return 0;
}