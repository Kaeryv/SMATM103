/**
 * \file cgradient.h
 * \date 10/12/18
 */

#ifndef EXAMTD_CGRADIENT_H
#define EXAMTD_CGRADIENT_H

#include "matrix.h"

/**
 * cgradient
 * \brief Conjugate gradient solver
 * \param A The matrix, should be column-major ordered.
 * \param b The rhs vector
 * \param x The initial guess (input), the solution (output).
 * \param path Where to dump the convergence profile vector
 * \return error code (0 is fine)
 */
int cgradient(Matrix A, Matrix b, Matrix x, const char *path);

/**
 * pcgradient
 * \brief Conjugate gradient solver with preconditionning
 * \param A The matrix, should be column-major ordered.
 * \param b The rhs vector
 * \param x The initial guess (input), the solution (output).
 * \param M The precondition matrix, should be column-major ordered.
 * \param path Where to dump the convergence profile vector
 * \return error code (0 is fine)
 */
int pcgradient(Matrix A, Matrix b, Matrix x, Matrix M, const char *path);

#endif //EXAMTD_CGRADIENT_H
