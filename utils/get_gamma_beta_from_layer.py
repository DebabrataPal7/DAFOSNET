#!/usr/bin/env python
# coding: utf-8

from models.Feature_extractor import CDFSOSR_Model
import tensorflow as tf

#get current gamma beta values from FE
def get_gamma_beta(CDFSOSR_Model):

    Gamma3_D1 = CDFSOSR_Model.get_layer('BN3_D1').get_weights()[0]
    Beta3_D1 = CDFSOSR_Model.get_layer('BN3_D1').get_weights()[1]
    Gamma3_D2 = CDFSOSR_Model.get_layer('BN3_D2').get_weights()[0]
    Beta3_D2 = CDFSOSR_Model.get_layer('BN3_D2').get_weights()[1]

    return Gamma3_D1, Gamma3_D2, Beta3_D1, Beta3_D2