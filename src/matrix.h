/**
 * @file matrix.h
 * @brief Matrix utilitaries definitions.
 * @author Nicolas Roy
 * @date 10/12/18
 */

#ifndef EXAMTD_MATRIX_H
#define EXAMTD_MATRIX_H

#include "kinds.h"
#include <stdbool.h>


/**
 * \brief Macro for accessing array elements
 */
#define AT(A, i, j) A.data[j*A.m+i]

enum loadable
{
    MATRIX, VECTOR, CHOLES
};

/**
 * \enum MatrixType
 * \brief Hints for the solver and other routines on the Matrix structure.
 */
enum MatrixType
{
    DENSE = 0, ///< Dense matrix, using LU Facto
    DIAGONAL,  ///< Diagonal matrix, using simple division
    SDP,       ///< Using dpsov from lapacke
    SYM,       ///< Using dstsv
    BAND       ///< Using dgbsv, using band storage
};


/**
 * @struct Matrix matrix.h
 * @brief Structure defining a matrix handle.
 */
typedef struct Matrix Matrix;
struct Matrix
{
    real *restrict data;     ///< Values pointer
    unsigned m;              ///< Number of rows
    unsigned n;              ///< Number of cols
    enum MatrixType type;    ///< Structure hint
    unsigned kl;             ///< Subdiagonals
    unsigned ku;             ///< Overdiagonals
    bool factored;
    int *restrict ipiv;
};


/**
 * @brief Allocates Matrix data.
 * @param A
 */

int allocate(Matrix *matrix);

/**
 * @brief Frees Matrix data.
 */
void deallocate(Matrix matrix);

/**
 * @brief Prints given Matrix to console.
 */
void print(Matrix matrix);

/**
 * \brief Prints given Matrix to file.
 */
void dump(Matrix matrix, const char *path);

/**
 * \brief
 * \param matrix
 * \param path
 * \alert Useless
 */
void dump_profile(Matrix matrix, const char *path);

/**
 *
 * \param A Left-hand side
 * \param B Right-hand side
 * \param C Solution container, already allocated.
 * \return Error code (fine if zero).
 */
int s_matmul(Matrix A, Matrix B, Matrix C);

/**
 * \brief Loads matrix from given file
 * \param path
 * \param L The file format (vector, matrix or cholesky precond).
 * \return The matrix handle
 */
Matrix load_from_file(const char *path, enum loadable L);

/**
 * \brief Brand new matrix from source
 * \param src Matrix to copy
 * \return Handle to new matrix
 */
Matrix copy(Matrix src);

/**
 * \brief Solve the system Ax=b
 * \param A The system, should contain hint of matrix structure
 * \param b
 * \return x
 */
Matrix solve(Matrix *A, Matrix b);

/**
 * \brief Solve without allocation
 * Defers the peration to solve+init.
 * \param A The system
 * \param b
 * \param x The already allocated return handle
 */
void ssolve(Matrix *A, Matrix b, Matrix x);

/**
 * \brief Convert DENSE matrix handle to BAND matrix handle.
 * \param A
 * \param KU
 * \param KL
 */
void band_store(Matrix *A, unsigned KU, unsigned KL);

#endif //EXAMTD_MATRIX_H
