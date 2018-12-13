//
// Created by kaeryv on 10/12/18.
//

#ifndef EXAMTD_CGRADIENT_H
#define EXAMTD_CGRADIENT_H

#include "matrix.h"

/**
 * cgradient
 *
 * \brief Computes conjugate gradient of given Matrix
 * \param A The matrix, should be column-major ordered.
 * \param b The rhs vector.
 * \param x The initial guess (input), the solution (output).
 * \author Nicolas Roy
 * \return Error code (fine if 0).
 */
int cgradient(Matrix A, Matrix b, Matrix x);

int pcgradient(Matrix A, Matrix b, Matrix x, Matrix M);

Matrix dpssolv(Matrix A, Matrix b);

#endif //EXAMTD_CGRADIENT_H
