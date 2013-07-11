from kanalyze.training.datasets import TrainingDatasetScreen2
import numpy as np
import itertools
import tabular as tb


class dataset1(TrainingDatasetScreen2):

    def __init__(self):
        TrainingDatasetScreen2.__init__(self, internal_canonical=True)

    def get_images(self, preproc=None, global_light_spec=None, get_models=True):
        if not preproc:
            preproc = {'size': (256, 256), 'mode': 'L', 'normalize': False, 'dtype': 'float32'}
        return super(dataset1, self).get_images(preproc, global_light_spec, get_models)

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

        :return: set of model names in second half of objects
        """
        meta = self.meta
        if not hasattr(self, '_obj_set_2'):
            self._obj_set2 = set(itertools.chain.from_iterable(
                [objs[1::2] for objs in [np.unique(meta['obj'][meta['category'] == cat])
                                         for cat in np.unique(meta['category'])]]))
        return self._obj_set2

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            try:
                WTF_TABULAR = tb.io.loadbinary(self.CACHE_PATH+'/dataset1_meta.npz')# This seems like a flaw with tabular's loadbinary.
                self._meta = tb.tabarray(records=WTF_TABULAR[0], dtype=WTF_TABULAR[1])
            except IOError:
                print self.CACHE_PATH+'/dataset1_meta.npz'
                self._meta = self._get_meta()
                tb.io.savebinary(self.CACHE_PATH+'/dataset1_meta.npz', self._meta)
        return self._meta
