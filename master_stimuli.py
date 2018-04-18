from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .passive_active_2 import PassAct2
from .passive_active_3 import PassAct3, PassAct3Aud, PassAct3AudVis


__all__ = ['MasterStimuli']


class MasterStimuli:

    def __init__(self):
        pass

    @staticmethod
    def paradigm_from_experiment(experiment):
        if experiment == 'PassAct2':
            return PassAct2()
        if experiment == 'PassAct3':
            return PassAct3()
        if experiment == 'PassAct3Aud':
            return PassAct3Aud()
        if experiment == 'PassAct3AudVis':
            return PassAct3AudVis()
        raise ValueError('Unknown experiment: {}'.format(experiment))

    @staticmethod
    def stimuli_from_name(stimuli_name):
        if stimuli_name == 'passive_active_2':
            return PassAct2().master_stimuli
        if stimuli_name == 'passive_active_3':
            return PassAct3().master_stimuli
        raise ValueError('Unknown stimuli set: {}'.format(stimuli_name))
