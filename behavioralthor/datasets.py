from kanalyze.training.datasets import TrainingDatasetScreen2
import cPickle
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

    @property
    def fruit_vs_chair_splits(self):

        """Saved splits for chair vs fruit category task"""
        if not hasattr(self, '_fruit_vs_chair_splits'):
            try:
                self._fruit_vs_chair_splits = cPickle.load(open(self.CACHE_PATH + 'fruits_vs_chair_splits', 'rb'))
            except IOError:
                npc_train = 20
                npc_test = [20, 20]
                num_splits = 5
                catfunc = lambda x: x['category']
                train_q = lambda x: (x['obj'] in self.obj_set1) and (x['category'] in ['Fruits', 'Chairs'])
                test_q = train_q
                _fruit_vs_chair_splits = self.get_subset_splits(self.meta,
                                                                npc_train,
                                                                npc_test,
                                                                num_splits,
                                                                catfunc,
                                                                train_q,
                                                                test_q)
                cPickle.dump(_fruit_vs_chair_splits, open(self.CACHE_PATH + 'fruits_vs_chair_splits', 'wb'))
                self._fruit_vs_chair_splits = _fruit_vs_chair_splits
        return self._fruit_vs_chair_splits
