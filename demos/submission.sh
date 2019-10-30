#!/usr/bin/env bash
# -*- coding: UTF-8 -*-

echo -e "************ start of ${0}\n"
set -x


## args parse
arch=${1-DispNetC}
maxdisp=${2-192}
nckp=${3-80}
flag_train=${4-"T(sf-tr)_F(k15-tr,k12-tr)"} # "T(sf-tr)"} # 
datas_val=${5-"k15-te"} # "k12-te"} # 
dir_datas_val=${6-"/media/qjc/D/data/kitti"}
bn=${7-1}
echo


## loadmodel
mode="Submission"
loadmodel="./results/Train_${arch}_${flag_train}/weight_${nckp}.pkl"


# log_filepath and dir_save
flag="${mode}_${datas_val}/${arch}_${flag_train}"
dir_save="./results/${flag}"
LOG="${dir_save}/log_`date +%Y-%m-%d_%H-%M-%S`.txt"
mkdir -p "${dir_save}"
echo


# val model
freq_print=1
./main.py --mode ${mode} --arch $arch --maxdisp $maxdisp \
               --loadmodel $loadmodel \
               --datas_val $datas_val --dir_datas_val $dir_datas_val  \
               --bn $bn\
               --freq_print $freq_print \
               --dir_save $dir_save \
               2>&1 | tee -a "$LOG"
echo


echo -e "************ end of ${0}\n\n\n"




