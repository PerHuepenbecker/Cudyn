#!/bin/bash

set -e

curl -L https://www.cise.ufl.edu/research/sparse/MM/Janna/ML_Laplace.tar.gz -o ML_Laplace.tar.gz

tar -xvzf ML_Laplace.tar.gz

curl -L https://www.cise.ufl.edu/research/sparse/MM/LAW/hollywood-2009.tar.gz -o hollywood-2009.tar.gz

tar -xvzf hollywood-2009.tar.gz

echo "Finished!"