/**
 * @file matrix.c
 * @brief Matrix utilitaries implementations.
 * @author Nicolas Roy
 * @date 10/12/18
 */

#include "matrix.h"

#include <cblas.h>
#include <stdlib.h>
#include <malloc.h>
#include <lapacke.h>


int allocate(Matrix *matrix)
{
  if (matrix->n != 0 && matrix->m != 0)
  {
    matrix->data = (real *) calloc(matrix->m * matrix->n, sizeof(real));
    if (matrix->data == NULL)
    {
      fprintf(stderr, "Matrix allocation failed.");
      exit(1);
    }
  }
  else
  {
    fprintf(stderr, "At least one matrix dimension is zero !");
  }
}


void deallocate(Matrix matrix)
{
  free(matrix.data);
}


real dot(Matrix A, Matrix B)
{
  real res = 0.;
  if (A.m == B.m)
  {
    for (int i = 0; i < A.m; ++i)
    {
      res += AT(A, i, 0) * AT(B, i, 0);
    }
    return res;
  }
  else
  {
    return 0;
  }
}

void print(Matrix matrix)
{
  for (int i = 0; i < matrix.m; ++i)
  {
    for (int j = 0; j < matrix.n; ++j)
    {
      printf("\t%lf", AT(matrix, i, j));
    }
    printf("\n");
  }
}

Matrix copy(Matrix src)
{
  Matrix result;
  result.m = src.m;
  result.n = src.n;
  result.type = src.type;
  result.kl = src.kl;
  result.ku = src.ku;
  allocate(&result);
  for (int i = 0; i < result.m * result.n; ++i)
    result.data[i] = src.data[i];
  return result;
}

int s_matmul(Matrix A, Matrix B, Matrix C)
{
  gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
       A.m, B.n, A.n, 1.0,
       A.data, A.m, B.data,
       B.m, 0.0, C.data, C.m);
  return 0;
}


Matrix matmul(Matrix A, Matrix B)
{
  Matrix C;
  C.m = A.m;
  C.n = B.n;
  allocate(&C);
  s_matmul(A, B, C);
  return C;
}

int s_matmul_transb(Matrix A, Matrix B, Matrix C)
{
  gemm(CblasColMajor, CblasNoTrans, CblasTrans,
       A.m, B.n, A.n, 1.0,
       A.data, A.m, B.data,
       B.m, 0.0, C.data, C.m);
  return 0;
}

Matrix load_from_file(const char *path, enum loadable L)
{
  FILE *file = fopen(path, "r");
  if (file == NULL)
  { fprintf(stderr, "Failed to open file %s.", path); }
  unsigned n = 0;
  fscanf(file, "%d", &n);
  if (n <= 0)
  { fprintf(stderr, "Failed to read matrix file."); }

  Matrix result = {.m=n, .n=n, .data=NULL};


  if (L == VECTOR)
  {
    result.n = 1;
  }
  else
  {
    n = n * n;
  }

  allocate(&result);

  for (int i = 0; i < n; ++i)
  {
    fscanf(file, "%lf", &result.data[i]);
  }
  return result;
}

void dump(Matrix matrix, const char *path)
{
  FILE *file = fopen(path, "w");
  for (int i = 0; i < matrix.m * matrix.n; ++i)
  {
    fprintf(file, "%le\n", matrix.data[i]);
  }
  fclose(file);
}

void dump_profile(Matrix matrix, const char *path)
{
  FILE *file = fopen(path, "w");
  for (int i = 0; i < matrix.m * matrix.n; ++i)
  {
    if (AT(matrix, i, 0) == 0.0)
    {
      break;
    }
    fprintf(file, "%d %lf\n", i, matrix.data[i]);
  }
  fclose(file);
}

Matrix solve(Matrix A, Matrix b)
{
  Matrix result = {.m=b.m, .n=1, .data=NULL};
  allocate(&result);
  ssolve(A, b, result);
  return result;
}

void ssolve(Matrix A, Matrix b, Matrix x)
{
  cblas_dcopy(b.m, b.data, 1, x.data, 1);
  if (A.type == DIAGONAL)
  {
    for_range(i, x.m)
    {
      AT(x, i, 0) /= AT(A, i, i);
    }
  }
  else
  {
    if (A.type == SDP)
    {
      posv(LAPACK_COL_MAJOR, 'L', A.m,
           x.n, A.data, A.n, x.data,
           x.m);
    }
    else
    {
      if (A.type == SYM)
      {
        Matrix tmp = copy(A);
        int *ipiv = calloc(A.m, sizeof(int));
        LAPACKE_dsysv(LAPACK_COL_MAJOR, 'L', A.m, x.n, tmp.data, A.n, ipiv, x.data, A.m);
        deallocate(tmp);
      }
      else
      {
        if (A.type == BAND)
        {
          Matrix AB = copy(A);
          // dgbsv stores other stuff in AB, be cautious !!!

          int *ipiv = calloc(A.n, sizeof(int));
          LAPACKE_dgbsv(LAPACK_COL_MAJOR, AB.n, A.kl, A.ku, 1, AB.data, AB.m, ipiv, x.data, x.m);
        }
        else
        {
          //printf("USING DGETRF\n");
          Matrix tmp = copy(A);
          int *ipiv = calloc(A.m, sizeof(int));
          LAPACKE_dgetrf(LAPACK_COL_MAJOR, A.m, A.n, tmp.data, A.n, ipiv);
          LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'n', A.m, 1, tmp.data, A.m, ipiv, x.data, A.m);
        }
      }
    }
  }
}

void band_store(Matrix *A, unsigned KU, unsigned KL)
{
  int K = KL + KU;
  Matrix AB = {.m=1 + 2 * KL + KU, .n=A->n, .data=NULL};

  allocate(&AB);
  for_range(j, A->n)
  {
    int imax = min(A->n, j + (int) KL + 1);
    int imin = max(0, j - (int) KU);

    for (int i = imin; i < imax; ++i)
    {
      AT(AB, K + i - j, j) = AT((*A), i, j);
    }
  }

  deallocate((*A));
  AB.kl = KL;
  AB.ku = KU;
  AB.type = BAND;
  *A = AB;
}
