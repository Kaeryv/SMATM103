//
// Created by kaeryv on 10/12/18.
//

#include <stdio.h>
#include <lapacke.h>
#include <cblas.h>
#include "precond.h"

Matrix precond_jacobi(Matrix A)
{
  Matrix ret = {.m=A.m, .n=A.n, .data=NULL};
  allocate(&ret);
  cblas_dscal(ret.m * ret.n, 0.0, ret.data, 1);
  for (int i = 0; i < A.m; ++i)
    AT(ret, i, i) = AT(A, i, i);
  ret.type = DIAGONAL;
  dump(ret, "M_JACOBI.txt");
  return ret;
}

Matrix precond_ssor(Matrix A, real omega)
{
  Matrix DL = copy(A);

  // D will also store the result
  Matrix D = {.m=A.m, .n=A.n, .data=NULL};
  allocate(&D);

  // We construct D
  for_range(i, A.m)
  {
    AT(D, i, i) = AT(A, i, i);
  }

  // Scale D by 1/omega
  cblas_dscal(D.m * D.n, 1.0 / omega, D.data, 1);

  // Set diagonal of DL = A to D diagonal
  for_range(i, D.m) AT(DL, i, i) = AT(D, i, i);


  // Invert diagonal D
  for_range(i, D.m)AT(D, i, i) = 1.0 / AT(D, i, i);

  // Now the lower triangle and diagonal of L are fine,
  // we do not care about the upper part.
  // D contains the inverse of D

  // We perform (D/omega+L)*inv(D)
  // Wich here is expressed D = DL*D
  // Wich cblas expresses as D := alpha*op( DL )*D op is notrans
  cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
              CblasNonUnit, D.m, D.n, 1.0, DL.data, A.m, D.data, A.n);

  // Now we have D = DL*D, we need to right-multiply D by DL'
  // We will use the same routine but right-side
  // D := alpha*D*op( DL ) op will be trans

  cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
              CblasNonUnit, D.m, D.n, 1. / (2. - omega), DL.data, A.m, D.data, A.n);


  // La matrice en sortie est SymDefPos
  D.type = SDP;
  dump(D, "M_SSOR.txt");
  return D;
}

Matrix precond_spectral(Matrix A, int order)
{
  Matrix eigenvectors = copy(A);
  Matrix eigenvalues = {.m=A.m, .n=1, .data=calloc(A.m, sizeof(real))};

  // First, we need to compute the eigenvalues and eigenvectors of A.
  int info =
          LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'L',
                        eigenvectors.m,
                        eigenvectors.data,
                        eigenvectors.n,
                        eigenvalues.data);

  if (info != 0)
  {
    fprintf(stderr, "precond_spectral did an oopsie. Errno: %d\n", info);
  }

  Matrix M = {.m=A.m, .n=A.n, .data=NULL};
  allocate(&M);

  // M = I_m + \sum{(\lambda_i-1)*v_iv_i^T}
  for_range(i, M.m)
  {
    AT(M, i, i) = 1.0;
    for_range(j, M.n)
    {
      for_range(k, 50)
      {
        AT(M, i, j) += (AT(eigenvalues, k, 0) - 1.0) * AT(eigenvectors, i, k) * AT(eigenvectors, j, k);
      }
    }
  }
  M.type = SYM;
  dump(M, "M_SPECTRAL.txt");
  return M;
}

Matrix compt_eigenvalues(Matrix A)
{
  Matrix result = {.m=A.m, .n=1, .data=NULL};
  allocate(&result);
  Matrix tmp = copy(A);
  LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'L',
                tmp.m,
                tmp.data,
                tmp.n,
                result.data);
  deallocate(tmp);
  return result;
}






