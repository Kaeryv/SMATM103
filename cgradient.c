//
// Created by kaeryv on 10/12/18.
//

#include "cgradient.h"

#include <stdio.h>
#include <math.h>
#include <lapacke.h>

int cgradient(Matrix A, Matrix b, Matrix x)
{
  // Check for consistency
  if (A.n != b.m) {
    fprintf(stderr, "Matrix sizes are not consistent for multiplication.");
    return 1;
  }
  // r0 = b - Ax0
  Matrix r = matmul(A, x);

  for (int i = 0; i < r.m; ++i)
    AT(r, i, 0) = AT(b, i, 0) - AT(r, i, 0);

  Matrix pk = copy(r);
  Matrix Apk = copy(pk);
  for (int k = 0; k < 1000; ++k) {
    real alpha = dot(r, r);

    s_matmul(A, pk, Apk);
    real tmp = dot(pk, Apk);

    printf("%lf\n", tmp);
    if (fabs(tmp) < 1e-15) {
      fprintf(stderr, "Execution of cgradient interrupted because pk is orthogonal to Apk.\n");
      return 2;
    }
    alpha /= tmp;

    real beta = dot(r, r);

    // x += ak*pk
    for (int i = 0; i < x.m; ++i) {
      AT(x, i, 0) += alpha * AT(pk, i, 0);
      AT(r, i, 0) -= alpha * AT(Apk, i, 0);
    }
    real criterion = dot(r, r);
    printf("Criterion: %lf\n", criterion);
    if (criterion < 1e-6) {
      printf("Found solution, returning.\n");
      return 0;
    }
    // b_{k+1} = r_{k+1}*r_{k+1}/r_k*r_k
    if (fabs(beta) < 1e-15) {
      fprintf(stderr, "Execution of cgradient interrupted because beta got too small.\n");
      return 2;
    }
    beta = dot(r, r) / beta;
    for (int i = 0; i < x.m; ++i) {
      AT(pk, i, 0) = AT(r, i, 0) + beta * AT(pk, i, 0);
    }
  }
  fprintf(stderr, "Cgradient failed to converge.\n");

  return 3;
}


int pcgradient(Matrix A, Matrix b, Matrix x, Matrix M)
{
  // Check for consistency
  if (A.n != b.m) {
    fprintf(stderr, "Matrix sizes are not consistent for multiplication.");
    return 1;
  }
  real bnorm = dot(b, b);
  // r0 = Ax0
  Matrix r = matmul(A, x);
  // r0 <- Ax0 - b <- r0 - b
  for (int i = 0; i < r.m; ++i)
    AT(r, i, 0) = AT(r, i, 0) - AT(b, i, 0);

  // We solve the pre-conditionning system
  Matrix y = dpssolv(M, r);

  Matrix pk = copy(y);

  for (int i = 0; i < pk.m; ++i)
    AT(pk, i, 0) = -AT(pk, i, 0);
  Matrix Apk = copy(pk);
  for (int k = 0; k < 1000; ++k) {
    real alpha = dot(r, y);

    s_matmul(A, pk, Apk);
    alpha /= dot(pk, Apk);

    real beta = dot(r, y);

    // x += ak*pk
    for (int i = 0; i < x.m; ++i) {
      AT(x, i, 0) += alpha * AT(pk, i, 0);
      AT(r, i, 0) += alpha * AT(Apk, i, 0);
    }

    real criterion = dot(r, r);
    printf("Criterion: %lf\n", criterion);
    if (criterion / bnorm <= 1e-6) {
      printf("Found solution, returning.\n");
      return 0;
    }

    deallocate(y);
    y = copy(r);
    LAPACKE_dposv(LAPACK_COL_MAJOR, 'L', M.m,
                  y.n, M.data, M.n, y.data,
                  y.m);

    beta = dot(r, y) / beta;
    for (int i = 0; i < x.m; ++i) {
      AT(pk, i, 0) = -AT(y, i, 0) + beta * AT(pk, i, 0);
    }
  }
  fprintf(stderr, "Cgradient failed to converge.\n");

  return 3;
}

Matrix dpssolv(Matrix M, Matrix y)
{
  Matrix x;
  x = copy(y);
  posv(LAPACK_COL_MAJOR, 'L', M.m,
       x.n, M.data, M.n, x.data,
       x.m);
  return x;
}



