/**
 * @file cgradient.c
 * @brief Implements cgradient.h
 * @author Nicolas Roy
 */

#include "cgradient.h"

#include <stdio.h>
#include <lapacke.h>
#include <cblas.h>


int cgradient(Matrix A, Matrix b, Matrix x, const char *path)
{
  Matrix profile = {.m=1000, .n=1, .data=calloc(1000, sizeof(real))};

  // Norm of b, used in convergence criterion calculation.
  real bnorm = cblas_dnrm2(A.m, b.data, 1);

  // r = b
  Matrix r = copy(b);

  //  r = A*x0 - r
  cblas_dsymv(CblasColMajor, CblasLower, A.m, 1.0, A.data, A.m, x.data, 1, -1.0, r.data, 1);

  Matrix pk = copy(r);
  cblas_dscal(A.m, -1.0, pk.data, 1);

  Matrix Apk = copy(pk);

  for_range(k, 1000)
  {
    // Apk = A*pk
    cblas_dsymv(CblasColMajor, CblasLower, A.m, 1.0, A.data, A.m, pk.data, 1, 0.0, Apk.data, 1);

    real alpha = -cblas_ddot(A.m, r.data, 1, pk.data, 1) / cblas_ddot(A.m, pk.data, 1, Apk.data, 1);

    // x += ak*pk
    cblas_daxpy(A.m, alpha, pk.data, 1, x.data, 1);

    // r=b
    cblas_dcopy(A.m, b.data, 1, r.data, 1);

    // r = Ax - r
    cblas_dsymv(CblasColMajor, CblasLower, A.m, 1.0, A.data, A.m, x.data, 1, -1.0, r.data, 1);

    real criterion = cblas_dnrm2(r.m, r.data, 1) / bnorm;
    AT(profile, k, 0) = criterion;
    //printf("Criterion: %lf\n", criterion);

    if (criterion <= 1e-6)
    {
      //printf("Found solution, returning.\n");
      dump_profile(profile, path);
      return 0;
    }

    real beta = cblas_ddot(r.m, r.data, 1, Apk.data, 1) / cblas_ddot(pk.m, pk.data, 1, Apk.data, 1);
    // pk = - r + beta * pk
    cblas_dscal(pk.m, beta, pk.data, 1);
    cblas_daxpy(pk.m, -1.0, r.data, 1, pk.data, 1);
  }

  fprintf(stderr, "pcgradient failed to converge.\n");
  dump_profile(profile, path);
  return 1;
}


int pcgradient(Matrix A, Matrix b, Matrix x, Matrix M, const char *path)
{
  Matrix profile = {.m=1000, .n=1, .data=calloc(1000, sizeof(real))};

  // Norm of b, used in convergence criterion calculation.
  real bnorm = cblas_dnrm2(A.m, b.data, 1);

  // r = b
  Matrix r = copy(b);

  //  r = A*x0 - r
  cblas_dsymv(CblasColMajor, CblasLower, A.m, 1.0, A.data, A.m, x.data, 1, -1.0, r.data, 1);

  // We solve the pre-conditionning system  M y = r
  Matrix y = solve(M, r);

  Matrix pk = copy(y);
  cblas_dscal(A.m, -1.0, pk.data, 1);

  Matrix Apk = copy(pk);

  for_range(k, 1000)
  {
    cblas_dsymv(CblasColMajor, CblasLower, A.m, 1.0, A.data, A.m, pk.data, 1, 0.0, Apk.data, 1);

    // Init beta to its future denominator
    real beta = cblas_ddot(r.m, r.data, 1, y.data, 1);
    real alpha = beta / cblas_ddot(pk.m, pk.data, 1, Apk.data, 1);

    // x += ak*pk
    cblas_daxpy(A.m, alpha, pk.data, 1, x.data, 1);
    cblas_daxpy(A.m, alpha, Apk.data, 1, r.data, 1);

    real criterion = cblas_dnrm2(r.m, r.data, 1) / bnorm;
    AT(profile, k, 0) = criterion;
    //printf("Criterion: %lf\n", criterion);

    if (criterion <= 1e-6)
    {
      //printf("Found solution, returning.\n");
      dump_profile(profile, path);
      return 0;
    }

    // Solve My=r
    ssolve(M, r, y);

    beta = cblas_ddot(r.m, r.data, 1, y.data, 1) / beta;

    cblas_dscal(pk.m, beta, pk.data, 1);
    cblas_daxpy(pk.m, -1.0, y.data, 1, pk.data, 1);
  }
  fprintf(stderr, "pcgradient failed to converge.\n");
  dump_profile(profile, path);
  return 3;
}



