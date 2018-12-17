//
// Created by kaeryv on 10/12/18.
//

#ifndef EXAMTD_KINDS_H
#define EXAMTD_KINDS_H

#define KIND_DOUBLE

/**
 * Defining precision used in matrices.
 * KIND_DOUBLE for double aka real(8) precision.
 * KIND_FLOAT for float aka real or real(4) precision.
 *
 * CBlas and Lapacke subroutines are re-routed accordingly.
 */

#ifdef KIND_DOUBLE
typedef double real;
#define gemm cblas_dgemm
#define trtri LAPACKE_dtrtri
#define posv LAPACKE_dposv
#else
typedef float real;
#define gemm cblas_sgemm
#endif

#endif //EXAMTD_KINDS_H

#define for_range(index, max) for(int index = 0; index < max; ++index)
#define max(x, y) ((x) >= (y)) ? (x) : (y)
#define min(x, y) ((x) <= (y)) ? (x) : (y)