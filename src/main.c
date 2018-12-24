#include <stdio.h>
#include <libnet.h>
#include <cblas.h>

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
  char profile_file[64] = "./profile.txt";
  char output_token[64] = "./eigenvalues.txt";
  char ccholesky_precond[64] = "./data/donneeChol1.dat";
  int cholesky_precond = -1;
  enum PreconditionMethod precond_method = NONE;

  {
    int i = 1;
    while (i < argc)
    {
      if (strcmp("--input", argv[i]) == 0)
      {
        strncpy(matrix_file, argv[++i], 64);
        matrix_file[63] = '\0';
      }
      else
      {
        if (strcmp("--precond", argv[i]) == 0)
        {
          precond_method = atoi(argv[++i]);
        }
        else
        {
          if (strcmp("--prof", argv[i]) == 0)
          {
            strncpy(profile_file, argv[++i], 64);
            profile_file[63] = '\0';
          }
          else
          {
            if (strcmp("--out", argv[i]) == 0)
            {
              strncpy(output_token, argv[++i], 64);
              output_token[63] = '\0';
            }
            else
            {
              if (strcmp("--chol", argv[i]) == 0)
              {
                cholesky_precond = atoi(argv[++i]);
              }
              else
              {
                if (strcmp("--help", argv[i]) == 0)
                {
                  printf("Usage:\n--input <filename>\nTo select preconditionner:\n");
                  PRECOND_HELP_ENTRY(NONE);
                  PRECOND_HELP_ENTRY(JACOBI);
                  PRECOND_HELP_ENTRY(SSOR);
                  PRECOND_HELP_ENTRY(CHOLESKY);
                  printf("Use --chol to specify nr. of the Cholesky precondition matrix.\n");
                  PRECOND_HELP_ENTRY(SPECTRAL);
                  exit(0);
                }
                else
                {
                  printf("Option %s not recognized.\n", argv[i]);
                }
              }
            }
          }
        }
      }
      ++i;
    }
  }


  Matrix A = load_from_file(matrix_file, MATRIX);
  A.type = DENSE;

  dump(compt_eigenvalues(A), output_token);
  Matrix b = load_from_file("./data/donneeRHS.dat", VECTOR);

  Matrix x = {.m=A.m, .n=1, .data=NULL};
  allocate(&x);
  for_range(i, x.m)AT(x, i, 0) = 0.0;

  Matrix M = {.m=A.m, .n=A.n, .data=NULL};
  switch (precond_method)
  {
    case NONE:
    {
      if (cgradient(A, b, x, profile_file))
      {
        fprintf(stderr, "An error occured, pcgradient failed.");
      }
      else
      {
        return 0;
      }
      break;
    }
    case JACOBI:
      M = precond_jacobi(A);
      break;
    case SSOR:
      M = precond_ssor(A, 1.0);
      break;
    case CHOLESKY:
    {
      if (cholesky_precond < 0)
      {
        fprintf(stderr, "Specify Cholesky preconditionner nr\n");
        return 1;
      }
      else
      {
        sprintf(ccholesky_precond, "./data/donneeChol%d.dat", cholesky_precond);
      }

      Matrix K = load_from_file(ccholesky_precond, MATRIX);
      // M = KK'
      M = copy(K);
      cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                  M.m, M.n, 1.0, K.data, M.m, M.data, M.n);

      // The precond is at least symmetric
      M.type = SYM;

      // Convert M to band storage for 2 and 3, use bandwidth accordingly
      if (cholesky_precond == 2)
      {
        band_store(&M, 1, 1);
      }
      else
      {
        if (cholesky_precond == 3)
        {
          band_store(&M, 4, 4);
        }
      }
    }
      break;
    case SPECTRAL:
      M = precond_spectral(A, 100);
      break;
  }
  A.type = DENSE;
  pcgradient(A, b, x, M, profile_file);

  dump(x, "x.txt");
  return 0;
}
