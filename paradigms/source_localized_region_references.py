from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from itertools import groupby


__all__ = ['SourceLocalizedRegionReferences', 'SourceLocalizedRegionReference', 'region_references',
           'coalesce_lobes', 'sort_regions', 'to_lobe_names']


class SourceLocalizedRegionReference(object):

    def __init__(self, region, lobe, location, notes, hemisphere=None, index_sort=None):
        self._region = region
        self._lobe = lobe
        self._location = location
        self._notes = notes
        self._hemisphere = hemisphere
        self._index_sort = index_sort

    @property
    def region(self):
        return self._region

    @property
    def lobe(self):
        return self._lobe

    @property
    def location(self):
        return self._location

    @property
    def notes(self):
        return self._notes

    @property
    def hemisphere(self):
        return self._hemisphere

    @property
    def region_with_hemisphere(self):
        return self._region + '-' + self._hemisphere

    @property
    def index_sort(self):
        return self._index_sort


# noinspection PyPep8
def _no_hemisphere_references():
    # based on Mariya Toneva's excellent document available here:
    # https://docs.google.com/document/d/1TtN0vLOLMMTM3iuZ84ZLhj62ILQTsrrhOuNyyuxIhpc/edit
    # that document is itself based on https://www.frontiersin.org/articles/10.3389/fnins.2012.00171/full
    # to view the regions see: https://ars.els-cdn.com/content/image/1-s2.0-S1053811906000437-gr1.jpg

    # the order below is used as the sort order from top to bottom

    # noinspection PyPep8
    references = (

        # Frontal lobe
        SourceLocalizedRegionReference(
            'superiorfrontal',
                'frontal',
                'Superior frontal gyrus',
                '1/3 of the frontal lobe. Involved in self-awareness, in coordination with the action '
                'of the sensory system'),
        SourceLocalizedRegionReference(
            'caudalmiddlefrontal',
                'frontal',
                'Parts of middle frontal gyrus',
                'Role in reorienting attention'),
        SourceLocalizedRegionReference(
            'rostralmiddlefrontal',
                'frontal',
                'Parts of middle frontal gyrus',
                'Role in reorienting attention'),
        SourceLocalizedRegionReference(
            'parsopercularis',
                'frontal',
                'Inferior frontal gyrus',
                'Broca\'s area is in the left one, language production and phonological processing'),
        SourceLocalizedRegionReference(
            'parsorbitalis',
                'frontal',
                'Inferior frontal gyrus',
                None),
        SourceLocalizedRegionReference(
            'parstriangularis',
                'frontal',
                'Inferior frontal gyrus',
                'semantic language processing'),
        SourceLocalizedRegionReference(
            'lateralorbitofrontal',
                'frontal',
                'Orbitofrontal cortex, lateral division',
                'Part of prefrontal cortex, connections to amygdala, temporal pole. Lateral orbitofrontal cortex '
                'anticipates choices and integrates prior with current information. '
                'https://www.nature.com/articles/ncomms14823'),
        SourceLocalizedRegionReference(
            'medialorbitofrontal',
                'frontal',
                'Orbitofrontal cortex, medial division',
                'Part of prefrontal cortex, connections to hippocampus, cingulate making stimulus-reward associations '
                'and with the reinforcement of behavior'),
        SourceLocalizedRegionReference(
            'frontalpole',
                'frontal',
                'Frontal pole',
                'Part of prefrontal cortex, Area that\'s evolved most in humans. '
                'Strategic processes in memory recall and other executive functions. '
                'Cognitive branching (Cognitive branching enables a previously running task to be maintained in a '
                'pending state for subsequent retrieval and execution upon completion of the ongoing one.)'),
        SourceLocalizedRegionReference(
            'paracentral',
                'frontal',
                'Paracentral lobule',
                'Controls motor and sensory innervations of the contralateral lower extremity (lower leg). '
                'It is also responsible for control of defecation and urination'),
        SourceLocalizedRegionReference(
            'precentral',
                'frontal',
                'Precentral gyrus',
                'Primary motor cortex'),

        # wikipedia
        SourceLocalizedRegionReference(
            'insula',
                'insular',
                'Insula',
                'Deep within lateral sulcus separating frontal from temporal and parietal. '
                'Involved in conciousness, emotion, homeostasis. '
                'Multimodal sensing and sensory binding'
        ),

        # Parietal lobe
        SourceLocalizedRegionReference(
            'postcentral',
                'parietal',
                'Postcentral gyrus',
                'Primary somatosensory cortex (primary receptive area for touch)'),
        SourceLocalizedRegionReference(
            'inferiorparietal',
                'parietal',
                'Inferior parietal lobule',
                'Strong connections to wernicke\'s and broca\'s areas. Important in reading, language processing'),
        SourceLocalizedRegionReference(
            'supramarginal',
                'parietal',
                'Supramarginal gyrus',
                'Includes temporoparietal junction. Language perception and processing'),
        SourceLocalizedRegionReference(
            'superiorparietal',
                'parietal',
                'Superior parietal lobule',
                'Involved in spatial orientation, receives sensory input from hand and visual input'),
        SourceLocalizedRegionReference(
            'precuneus',
                'parietal',
                'Precuneus',
                'Mental imagery, episodic memory, self-awareness'),

        # Occipital lobe
        SourceLocalizedRegionReference(
            'cuneus',
                'occipital',
                'Cuneus',
                'Receives visual information from same side retina. Basic visual processing'),
        SourceLocalizedRegionReference(
            'lateraloccipital',
                'occipital',
                'Lateral occipital cortex (LOC in literature)',
                'Large role in object recognition'),
        SourceLocalizedRegionReference(
            'lingual',
                'occipital',
                'Lingual gyrus',
                'Visual processing of letters. '
                'Identification and recognition of words. '
                'Semantic word processing. '
                'Encoding of visual memories'),
        SourceLocalizedRegionReference(
            'pericalcarine',
                'occipital',
                'Pericalcarine cortex',
                'Primary visual cortex'),
        SourceLocalizedRegionReference(
            'isthmuscingulate',
                'occipital',
                'Isthmus of the cingulate gyrus',
                'Part of the cingulate cortex, which is an important part of the limbic system and is involved in '
                'emotion formation, learning, and memory. '
                'Connects cingulate cortex to parahippocampus'),
        SourceLocalizedRegionReference(
            'posteriorcingulate',
                'occipital',
                'Posterior cingulate cortex',
                'Part of the cingulate cortex, which is an important part of the limbic system and is involved in '
                'emotion formation, learning, and memory. '
                'Central node in the default mode network. '
                'Self-awareness, episodic memory retrieval'),
        SourceLocalizedRegionReference(
            'caudalanteriorcingulate',
                'occipital',
                'Part of the anterior cingulate cortex',
                'Part of the cingulate cortex, which is an important part of the limbic system and is involved in '
                'emotion formation, learning, and memory. '
                'Role in autonomic functions, like regulating blood pressure and heart rate. '
                'Also involved in error detection, anticipation of tasks, attention'),
        SourceLocalizedRegionReference(
            'rostralanteriorcingulate',
                'occipital',
                'Part of the anterior cingulate cortex',
                'Part of the cingulate cortex, which is an important part of the limbic system and is involved in '
                'emotion formation, learning, and memory. '
                'Role in autonomic functions, like regulating blood pressure and heart rate. '
                'Also involved in error detection, anticipation of tasks, attention'),

        # Medial temporal lobe
        SourceLocalizedRegionReference(
            'entorhinal',
                'medial-temporal',
                'Entorhinal cortex',
                'Medial-temporal structures are generally vital for declarative or long-term memory. '
                'Entorhinal: Memory and navigation; interface between hippocampus and neocortex'),
        SourceLocalizedRegionReference(
            'parahippocampal',
                'medial-temporal',
                'Parahippocampal gyrus (parahippocampal cortex includes fusiform)',
                'Medial-temporal structures are generally vital for declarative or long-term memory. '
                'Parahippocampal: Memory encoding and retrieval, scene recognition (parahippocampal place area)'),
        SourceLocalizedRegionReference(
            'temporalpole',
                'medial-temporal',
                'Anterior most portion of the temporal lobe (ATL)',
                'Medial-temporal structures are generally vital for declarative or long-term memory. '
                'Temporal pole: Critical for semantic memory'),
        SourceLocalizedRegionReference(
            'fusiform',
                'medial-temporal',
                'Fusiform gyrus',
                'Medial-temporal structures are generally vital for declarative or long-term memory. '
                'Portion of the left fusiform is thought to be related to word recognition (visual word form area). '
                'Fusiform face area (within-category identification)'),

        # Lateral temporal lobe
        SourceLocalizedRegionReference(
            'superiortemporal',
                'lateral-temporal',
                'Superior temporal gyrus',
                'Contains parts of primary auditory cortex and Wernicke\'s area. Important in language comprehension'),
        SourceLocalizedRegionReference(
            'inferiortemporal',
                'lateral-temporal',
                'Inferior temporal gyrus',
                'Processing, perception, and recognition of visual stimuli (especially objects). '
                'Part of the "what" stream of vision (identifying objects). '
                'In language, some evidence for place of syntactic unification during sentence processing'),
        SourceLocalizedRegionReference(
            'middletemporal',
                'lateral-temporal',
                'Middle temporal gyrus',
                'Accessing word meaning while reading'),
        SourceLocalizedRegionReference(
            'transversetemporal',
                'lateral-temporal',
                'Transverse temporal gyrus',
                'First cortical structure to process auditory info -- contains parts of primary auditory cortex'),
        SourceLocalizedRegionReference(
            'bankssts',
                'lateral-temporal',
                'Banks of superior temporal sulcus',
                'Projects to premotor areas. '
                'The auditory and multimodal sections in the upper bank of the STS (TAa and TPO) are reciprocally '
                'connected with the frontal cortex. '
                '(http://knightlab.berkeley.edu/statics/publications/2011/06/06/Hein_Knight_JOCN_20081.pdf)'),

        # other
        SourceLocalizedRegionReference(
            'corpuscallosum',
                'other',
                'Corpuscallosum',
                'Enables communication between hemispheres')
    )

    return references


class SourceLocalizedRegionReferences(object):

    def __init__(self):
        def _set_hemisphere(reference, hemisphere, index_sort):
            return SourceLocalizedRegionReference(
                reference.region, reference.lobe, reference.location, reference.notes, hemisphere, index_sort)

        undivided = _no_hemisphere_references()
        self._references = dict()
        for index, r in enumerate(undivided):
            left = _set_hemisphere(r, 'lh', index)
            right = _set_hemisphere(r, 'rh', index + len(undivided))
            self._references[left.region_with_hemisphere] = left
            self._references[right.region_with_hemisphere] = right

    def __getitem__(self, item):
        return self._references[item]

    def keys(self):
        return self._references.keys()


region_references = SourceLocalizedRegionReferences()


def coalesce_lobes(lobe_names):
    names = list()
    starts = list()
    index = 0
    for lobe_name, grouped in groupby(lobe_names):
        names.append(lobe_name)
        starts.append(index)
        for _ in grouped:
            index += 1
    return names, np.array(starts)


def to_lobe_names(region_names):
    return [region_references[name].lobe + '-' + region_references[name].hemisphere for name in region_names]


def sort_regions(region_names):
    sort_indices = np.argsort(np.array([region_references[name].index_sort for name in region_names]))
    region_names = list(sorted(region_names, key=lambda name: region_references[name].index_sort))
    return region_names, sort_indices
