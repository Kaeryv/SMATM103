/**
 * @file matrix.c
 * @brief Matrix utils implementations.
 * @author Nicolas Roy
 * @date 10/12/18
 */

#include "matrix.h"

#include <cblas.h>
#include <stdlib.h>
#include <malloc.h>


int allocate(Matrix *matrix)
{
  if (matrix->n != 0 && matrix->m != 0) {
    matrix->data = (real *) calloc(matrix->m * matrix->n, sizeof(real));
    if (matrix->data == NULL) {
      fprintf(stderr, "Matrix allocation failed.");
      exit(1);
    }
  } else
    fprintf(stderr, "At least one matrix dimension is zero !");
}


void deallocate(Matrix matrix)
{
  free(matrix.data);
}


real dot(Matrix A, Matrix B)
{
  real res = 0.;
  if (A.m == B.m) {
    for (int i = 0; i < A.m; ++i) {
      res += AT(A, i, 0) * AT(B, i, 0);
    }
    return res;
  } else {
    return 0;
  }
}

void print(Matrix matrix)
{
  for (int i = 0; i < matrix.m; ++i) {
    for (int j = 0; j < matrix.n; ++j) {
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
  if (file == NULL) fprintf(stderr, "Failed to open file %s.", path);
  unsigned n = 0;
  fscanf(file, "%d", &n);
  if (n <= 0) fprintf(stderr, "Failed to read matrix file.");

  Matrix result = {.m=n, .n=n, .data=NULL};


  if (L == VECTOR)
    result.n = 1;
  else
    n = n * n;

  allocate(&result);

  for (int i = 0; i < n; ++i) {
    fscanf(file, "%lf", &result.data[i]);
  }
  return result;
}

void dump(Matrix matrix, const char *path)
{
  FILE *file = fopen(path, "w");
  for (int i = 0; i < matrix.m * matrix.n; ++i) {
    fprintf(file, "%le\n", matrix.data[i]);
  }
  fclose(file);
}
