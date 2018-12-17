#!/bin/bash

EXECTBL=./cmake-build-debug/EXAMTD

for syst in 1 2 3
    do
    for precond in 0 1 2 4
    do
        echo "profile.$syst.$precond.txt"
        $EXECTBL --input data/donneeSyst$syst.dat --precond $precond --prof out/syst$syst.prec$precond.profile.txt --out out/eigenvals$syst.txt
    done

    precond=3
    $EXECTBL --input data/donneeSyst$syst.dat --precond $precond --prof out/syst$syst.prec$precond.profile.txt --out out/eigenvals$syst.txt --chol $syst
done