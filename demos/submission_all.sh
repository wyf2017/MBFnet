#!/usr/bin/env bash
# -*- coding: UTF-8 -*-

echo -e "************ start of ${0} ${*} ************\n"
set -x

maxdisp=192
nckp=10
# "T(sf-tr)_F(k15-tr,k12-tr)" # "T(sf-tr)" # "T(k15-tr,k12-tr)" # 
flag_train="T(sf-tr)_F(k15-tr,k12-tr)_F(k15,k12)" 
datas_val="k12-te" # "k15-te" # 
dir_datas_val="/media/qjc/D/data/kitti"
bn=1
kargs="${maxdisp} ${nckp} ${flag_train} ${datas_val} ${dir_datas_val} ${bn} "
entry="./demos/submission.sh"
echo

#${entry} "DispNetC" ${kargs}
#${entry} "DispNetS" ${kargs}

#${entry} "zMBFnet_S4B3W" ${kargs}
#${entry} "zMBFnet_S5B3W" ${kargs}
#${entry} "zMBFnet_S6B3W" ${kargs}
#${entry} "zMBFnet_S7B3W" ${kargs}

#${entry} "zMBFnet_S4B1W" ${kargs}
#${entry} "zMBFnet_S4B2W" ${kargs}
#${entry} "zMBFnet_S4B3W" ${kargs}
#${entry} "zMBFnet_S4B4W" ${kargs}
#${entry} "zMBFnet_S4B5W" ${kargs}
#${entry} "zMBFnet_S4B6W" ${kargs}

#${entry} "zMBFnet_S5B1W" ${kargs}
#${entry} "zMBFnet_S5B2W" ${kargs}
#${entry} "zMBFnet_S5B3W" ${kargs}
#${entry} "zMBFnet_S5B4W" ${kargs}
#${entry} "zMBFnet_S5B5W" ${kargs}


echo -e "************ end of ${0} ${*} ************\n\n\n"


