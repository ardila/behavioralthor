import behavioralthor
import imagenet.dldatasets
import cPickle
a = imagenet.dldatasets.HvM_Categories()
results = behavioralthor.utils.compute_all_2_ways_in_field(a.get_pixel_features(), a, n_jobs=8)
cPickle.dump(results, open('HvMCategoryPixel2WayResults'))
