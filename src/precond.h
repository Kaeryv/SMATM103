//
// Created by kaeryv on 10/12/18.
//

#ifndef EXAMTD_PRECOND_H
#define EXAMTD_PRECOND_H

#include "matrix.h"

Matrix precond_jacobi(Matrix A);

Matrix precond_ssor(Matrix A, real omega);

Matrix precond_spectral(Matrix A, int order);

Matrix compt_eigenvalues(Matrix A);

#endif //EXAMTD_PRECOND_H
