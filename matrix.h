//
// Created by kaeryv on 10/12/18.
//

#ifndef EXAMTD_MATRIX_H
#define EXAMTD_MATRIX_H

#include "kinds.h"

#define AT(A, i, j) A.data[j*A.m+i]

enum loadable
{
    MATRIX, VECTOR, CHOLES
};

typedef struct Matrix Matrix;
struct Matrix
{
    real *restrict data;
    unsigned m;
    unsigned n;
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

void dump(Matrix matrix, const char *path);


/**
 *
 * @param A
 * @param B
 * @return A*B
 */
Matrix matmul(Matrix A, Matrix B);

/**
 *
 * @param A
 * @param B
 * @param C Solution container.
 * @return Error code (fine if zero).
 */
int s_matmul(Matrix A, Matrix B, Matrix C);

int s_matmul_transb(Matrix A, Matrix B, Matrix C);

Matrix load_from_file(const char *path, enum loadable L);

Matrix copy(Matrix src);

real dot(Matrix A, Matrix B);

#endif //EXAMTD_MATRIX_H
