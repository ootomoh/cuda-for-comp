#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=0:10:00
#$ -N cof-comp
#$ -j y
. /etc/profile.d/modules.sh
module load cuda

./for-comp.out
