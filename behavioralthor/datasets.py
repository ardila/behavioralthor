from kanalyze.training.datasets import TrainingDatasetScreen2
import numpy as np
import itertools


class dataset1(TrainingDatasetScreen2):
    internal_canonical = True

    def get_images(self, preproc=None, global_light_spec=None, get_models=True):
        if not preproc:
            preproc = {'size': (256, 256), 'mode': 'L', 'normalize': False, 'dtype': 'float32'}
        super(dataset1, self).get_images(preproc, global_light_spec, get_models)

    @property
    def obj_set1(self):
        """
        First set of objects with every other object taken from each category

        :return: set of model names in first half of objects
        """
        meta = self.meta
        if not hasattr(self, '_obj_set_1'):
            self._obj_set1 = set(itertools.chain.from_iterable(
                [objs[0::2] for objs in [np.unique(meta['obj'][meta['category'] == cat])
                                         for cat in np.unique(meta['category'])]]))
        return self._obj_set1

    @property
    def obj_set2(self):
        """
        First set of objects with every other object taken from each category

        :return: set of model names in first half of objects
        """
        meta = self.meta
        if not hasattr(self, '_obj_set_2'):
            self._obj_set2 = set(itertools.chain.from_iterable(
                [objs[1::2] for objs in [np.unique(meta['obj'][meta['category'] == cat])
                                         for cat in np.unique(meta['category'])]]))
        return self._obj_set2
