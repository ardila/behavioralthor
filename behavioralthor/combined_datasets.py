import dldata
import imagenet
import dldata.stimulus_sets.dataset_templates as dt
import dldata.stimulus_sets.synthetic.synthetic_datasets as sd
import cPickle
import os


class LargeCombinedDataset(dt.CombinedDataset):
    def __init__(self):
        super(LargeCombinedDataset, self).init(
            [imagenet.dldatasets.ChallengeSynsets2013_offline, sd.RoschDataset],
            data=[None, None], aggregate_meta={'choose':[['synset', 'obj']], 'names': ['synset_obj']})

    def get_meta(self):
        path_to_meta = os.path.join(self.home(), 'meta'))
        if os.path.exists(path_to_meta):
            meta = cPickle.load(open(path_to_meta, 'rb'))
        else:
            meta = super(LargeCombinedDataset, self).get_meta()
            cPickle.dump(meta, open('large_combined_meta', 'wb'))
        return meta
