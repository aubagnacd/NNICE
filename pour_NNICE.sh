# CentOS version

# Env 
#unlimit
#
module purge
# CentOS version
export OS_VER2=`lsb_release -r | awk -F ":" '{print $2}'`
export OS_VER2=`echo $OS_VER2 | awk -F "." '{print $1}'`

if [ ! [$?LD_LIBRARY_PATH] ] 
then
   export LD_LIBRARY_PATH=""
fi

# Env 
#unlimit
#
module purge
# Configuration de compilation
#module load gimkl/2018b
if [ $OS_VER2 -gt 8 ]
then
   module use /soft/irsrvsoft1/expl/eb/centos_7/modules/all/Core
fi
module load GCC/9.3.0
module load GCCcore/9.3.0
module load intel/2020a
# Configuration de compilation
#module load gimkl/2018b
export CVG_LIBDIR=/home/irsrvshare2/R10/VKM-CONVERGE/GIT/LIB/V3.0/centos7_foss2019a/

export HDF5_DIR="/home/irsrvshare2/R10/VKM-CONVERGE/GIT/LIB/V3.0/centos7_foss2019a/HDF5-1.12.1_intel/"

# Initialisation des variables de compilation
export CC=gcc 
export CXX=g++ 

export HDF5_INC=${CVG_LIBDIR}/HDF5-1.12.1_intel/include
export HDF5_LIB=${CVG_LIBDIR}/HDF5-1.12.1_intel/lib
export EIGEN_INC=${CVG_LIBDIR}/eigen_3.4.0_intel/include/eigen3
export JSON_INC=${CVG_LIBDIR}/RapidJSON_intel/include
export NNICE_INC=$PWD
export NNICE_LIB=$PWD


## Repertoire d'install
## Includes
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CARNOT_LIB:$HDF5_LIB


