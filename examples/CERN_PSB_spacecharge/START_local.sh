#!/bin/bash

set -o errexit

if [ ! -n "$1" ]
  then
    echo "Usage: `basename $0` <name of the SC script> <N CPUs>"
    exit $E_BADARGS
fi

if [ ! -n "$2" ]
  then
    echo "Usage: `basename $0` <name of the SC script> <N CPUs>"
    exit $E_BADARGS
fi

source ../../customEnvironment.sh
echo "customEnvironment done"
source ../../../virtualenvs/py2.7/bin/activate
echo "python packages charged"
source /cvmfs/projects.cern.ch/intelsw/psxe/linux/all-setup.sh
echo "ifort charged (necessary for running)"

mpirun -np $2 ${ORBIT_ROOT}/bin/pyORBIT $1