//
// Created by kaeryv on 10/12/18.
//

#ifndef EXAMTD_PRECOND_H
#define EXAMTD_PRECOND_H

#include "matrix.h"

Matrix precond_jacobi(Matrix A);

Matrix precond_ssor(Matrix A);

Matrix precond_spectral(Matrix A);

void compt_eigenspace(Matrix A, Matrix eigenvectors, Matrix eigenvalues);

#endif //EXAMTD_PRECOND_H
