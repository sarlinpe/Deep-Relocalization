#!/bin/bash

#output="-oo $(pwd)/out.txt"
output="-I"
cores="15"
memory="4500"  # per core
scratch="5000"
gpus="1"
clock="4:00"
model="GeForceGTX1080Ti"
warn="-wt 15 -wa INT"

cmd="bsub
    -n $cores
    -W $clock $output
    $warn
    -R 'select[gpu_model0 == $model] rusage[mem=$memory,scratch=$scratch,ngpus_excl_p=$gpus]'
    $*"
echo $cmd
eval $cmd

# best: bsub -n1-R " select[gpu_model0 == 'GeForceGTX1080’|| gpu_model1==       'GeForceGTX1080’||gpu_model2 == 'GeForceGTX1080’||gpu_model3 ==     'GeForceGTX1080’] rusage[ngpus_excl_p=1]”
# see: https://github.com/prelz/BLAH/blob/master/src/scripts/lsf_submit.sh
# check names with lsload -s | grep gpu_model
