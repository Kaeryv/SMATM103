//
// Created by kaeryv on 10/12/18.
//

#include <stdio.h>
#include <lapacke.h>
#include "precond.h"

Matrix precond_jacobi(Matrix A)
{
  Matrix ret = {.m=A.m, .n=A.n, .data=NULL};
  allocate(&ret);

  for (int i = 0; i < A.m; ++i)
    AT(ret, i, i) = AT(A, i, i);

  return ret;
}

Matrix precond_ssor(Matrix A)
{
  real omega = 1.0;
  real iomega = 1.0 / omega;
  Matrix result = copy(A);
  Matrix DL = copy(A);
  Matrix D = A;
  allocate(&D);
  for (int i = 0; i < A.m; ++i) {
    // Inverse of diagonal matrix times 1/omega.
    AT(D, i, i) = omega / AT(A, i, i);
    AT(DL, i, i) = AT(DL, i, i) / omega;

    for (int j = i + 1; j < A.n; ++j) {
      AT(DL, i, j) = 0.;
    }
  }
  s_matmul_transb(D, DL, result);
  Matrix temp = copy(result);
  s_matmul(DL, temp, result);
  for_range(i, A.m * A.n) {
    result.data[i] *= 1. / (2 - omega);
  }
  return result;
}

Matrix precond_spectral(Matrix A)
{
  Matrix eigenvectors = copy(A);
  Matrix eigenvalues = {.m=A.m, .n=1, .data=calloc(A.m, sizeof(real))};

  // First, we need to compute the eigenvalues and eigenvectors of A.
  int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'L',
                           eigenvectors.m,
                           eigenvectors.data,
                           eigenvectors.n,
                           eigenvalues.data);

  if (info != 0)
    fprintf(stderr, "precond_spectral did an oopsie. Errno: %d\n", info);

  Matrix M = {.m=A.m, .n=A.n, .data=NULL};
  allocate(&M);

  // We init the precond matrix to eye(m)
  for_range(i, M.m) {
    AT(M, i, i) = 1.0;
  }

  // M = I_m + \sum{(\lambda_i-1)*v_iv_i^T}
  // hence M += \sum{(\lambda_i-1)*v_iv_i^T}
  for_range(k, M.m) {
    for_range(i, M.m) {
      for_range(j, M.n) {
        AT(M, i, j) += (AT(eigenvalues, k, 0) - 1.0) * AT(eigenvectors, i, k) * AT(eigenvectors, j, k);
      }
    }
  }
  return M;
}






