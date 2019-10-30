#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import os
sys.path.append(os.path.dirname(__file__))


def model_by_name(name, maxdisp):
    
    # DispNet[S/C]
    if('DispNet'.lower() in name.lower()):
        from .DispNet import get_model_by_name
        model = get_model_by_name(name, maxdisp)

    # zMBFnet
    elif('zMBFnet'.lower() in name.lower()):
        from .zMBFnet import get_model_by_name
        model = get_model_by_name(name, maxdisp)

    # Unsupported model
    else:
        raise Exception('Unsupported model: ' + name)
    
    return model



