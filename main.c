#include <stdio.h>
#include <libnet.h>

#include "cgradient.h"
#include "precond.h"

#define PRECOND_HELP_ENTRY(token) printf("--precond %d for %s.\n", token, #token)

enum PreconditionMethod
{
    NONE, JACOBI, SSOR, CHOLESKY, SPECTRAL
};

int main(int argc, char **argv)
{
  // Reading input arguments
  char matrix_file[64] = "./data/donneeSyst1.dat";
  enum PreconditionMethod precond_method = NONE;

  {
    int i = 1;
    while (i < argc) {
      if (strcmp("--input", argv[i]) == 0) {
        strncpy(matrix_file, argv[++i], 64);
        matrix_file[63] = '\0';
      } else if (strcmp("--precond", argv[i]) == 0) {
        precond_method = atoi(argv[++i]);
      } else if (strcmp("--help", argv[i]) == 0) {
        printf("Usage:\n--input <filename>\nTo select preconditionner:\n");
        PRECOND_HELP_ENTRY(NONE);
        PRECOND_HELP_ENTRY(JACOBI);
        PRECOND_HELP_ENTRY(SSOR);
        PRECOND_HELP_ENTRY(CHOLESKY);
        PRECOND_HELP_ENTRY(SPECTRAL);
        exit(0);
      } else {
        printf("Option %s not recognized.\n", argv[i]);
      }
      ++i;
    }
  }


  Matrix A = load_from_file(matrix_file, MATRIX);
  Matrix b = load_from_file("./data/donneeRHS.dat", VECTOR);

  Matrix M = {.m=A.m, .n=A.n, .data=NULL};
  switch (precond_method) {
    case NONE:
      allocate(&M);
      for_range(i, A.m) AT(M, i, i) = 1.0;
      break;
    case JACOBI:
      M = precond_jacobi(A);
      break;
    case CHOLESKY:
      break;
    case SSOR:
      M = precond_ssor(A);
      break;
    case SPECTRAL:
      M = precond_spectral(A);
      break;
  }

  Matrix x = {.m=A.m, .n=1, .data=NULL};
  allocate(&x);

  for (int i = 0; i < M.m; ++i) {
    for (int j = 0; j < i; ++j) {
      AT(M, j, i) = AT(M, i, j);
    }
    //AT(M, i, i) = 1. / AT(M, i, i);
    AT(x, i, 0) = 1.0;
  }
  dump(M, "M.txt");


  if (pcgradient(A, b, x, M)) {
    fprintf(stderr, "An error occured, pcgradient failed.");
  }
  FILE *out = fopen("./resultSys1.txt", "w+");
  for (int i = 0; i < 100; ++i) {
    fprintf(out, "%lf\n", x.data[i]);
  }
  printf("First element: %lf\n", x.data[0]);
  printf("Last element: %lf\n", x.data[x.m - 1]);
  fclose(out);

  return 0;
}