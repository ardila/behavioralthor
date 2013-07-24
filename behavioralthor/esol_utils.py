import json
import csv
import re
import scipy.io
import scipy.stats
import numpy as np
import os
import itertools
import cPickle as pk
import pymongo
import pylab as plt

def SBcorrection(corr, mult_factor):
    pred = (mult_factor*corr)/(1+(mult_factor-1)*corr)
    return pred

def normalize_CM(CF):
    new_CF = np.zeros(np.shape(CF))
    for col in range(0, np.shape(CF)[1]):
        total = np.sum(CF[:,col])
        norm_col = CF[:,col]/float(total)
        new_CF[:,col] = norm_col
    return new_CF

def offDmass(CF):
    return sum(CF[np.eye(CF.shape[0])==0]/float(sum(CF)))

class expDataDB(object):
    
    def __init__(self, collection, selector, numObjs, models = None):
        
        if models == None:
            self.models = ['weimaraner', 'lo_poly_animal_TRTL_B', 'lo_poly_animal_ELE_AS1',
                         'lo_poly_animal_TRANTULA', 'foreign_cat', 'lo_poly_animal_CHICKDEE',
                         'lo_poly_animal_HRS_ARBN', 'MB29346', 'MB31620', 'MB29874',
                         'interior_details_033_2', 'MB29822', 'face7', 'single_pineapple',
                         'pumpkin_3', 'Hanger_02', 'MB31188', 'antique_furniture_item_18',
                         'MB27346', 'interior_details_047_1', 'laptop01', 'womens_stockings_01M',
                         'pear_obj_2', 'household_aid_29', '22_acoustic_guitar', 'MB30850',
                         'MB30798', 'MB31015', 'Nurse_pose01', 'fast_food_23_1', 'kitchen_equipment_knife2',
                         'flarenut_spanner', 'womens_halterneck_06', 'dromedary', 'MB30758',
                         'MB30071', 'leaves16', 'lo_poly_animal_DUCK', '31_african_drums',
                         'lo_poly_animal_RHINO_2', 'lo_poly_animal_ANT_RED', 'interior_details_103_2',
                         'interior_details_103_4', 'MB27780', 'MB27585', 'build51', 'Colored_shirt_03M',
                         'calc01', 'Doctor_pose02', 'bullfrog', 'MB28699', 'jewelry_29', 'trousers_03',
                         '04_piano', 'womens_shorts_01M', 'womens_Skirt_02M', 'lo_poly_animal_TIGER_B',
                         'MB31405', 'MB30203', 'zebra', 'lo_poly_animal_BEAR_BLK', 'lo_poly_animal_RB_TROUT',
                         'interior_details_130_2', 'Tie_06']
        
        else:
            self.models = models
        
        models_idxs = {}
        for idx, model in enumerate(self.models):
            models_idxs[model] = idx
        self.models_idxs = models_idxs
        
        conn = pymongo.Connection(port = 22334, host = 'localhost')
        db = conn.mturk
        col = db[collection]
        
        self.subj_data = col.find(selector, {'ImgData':1, 'Response':1, 'StimShown':1})
        self.trial_data = self.preprocess(self.subj_data)
        self.numResp = len(self.trial_data[0][2])
        self.numObjs = numObjs
        self.totalTrials = len(self.trial_data)
        self.corr_type = 'pearson'
        
    def init_from_pickle(self, pkFile):
        f = open(pkFile, 'rb')
        data = pk.load(f)
        f.close()
        self.subj_data = data
        self.trial_data = self.preprocess(self.subj_data)
        self.totalTrials = len(self.trial_data)
        
    def setPopCM(self):
        if self.numResp == 2:
            self.popCM, self.CM_order = self.getPopCM2x2fast(self.trial_data)
        else:
            self.popCM, self.CM_order = self.getPopCM(self.trial_data)
    
    def preprocess(self, subj_data):
        RV = [] #Response vector
        SV = [] #Stimulus vector
        DV = [] #Distractor vector
        for subj in subj_data:
            for r in subj['Response']:
                RV.append(r.split('/')[-1].split('.')[0][:-6])
            for s in subj['StimShown']:
                DV.append( [ s_.split('/')[-1].split('.')[0] for s_ in s if s_ != s[0] ] )
            for i in subj['ImgData']:
                SV.append(str(i[0]['obj']))
        new_data = []
        for idx, shown in enumerate(SV):
            model = shown
            CF_col_idx = self.models_idxs[model]
            CF_row_idx = self.models_idxs[RV[idx]]
            new_data.append([CF_col_idx, CF_row_idx, [self.models_idxs[m] for m in DV[idx]]]) #order is shown, picked, distractors
        return new_data
    
    def getPopCM2x2fast(self, trial_data, subsel = None):
        combs = list(itertools.combinations(range(0, self.numObjs), 2))
        CMs = {}
        for c in combs:
            CMs[c] = np.zeros((2,2))
        for t in trial_data:
            target = t[0]
            pick = t[1]
            cm = tuple(sorted(t[2])) #Because itertools always spits out the combs in sorted order
            if target == cm[0]:
                if target == pick:
                    CMs[cm][0,0] += 1
                else:
                    CMs[cm][1,0] += 1
            else:
                if target == pick:
                    CMs[cm][1,1] += 1
                else:
                    CMs[cm][0,1] += 1
        if subsel is None:
            return [CMs[c] for c in combs], combs
        else:
            return [CMs[c] for c in subsel], subsel
    
    def getPopCM(self, trial_data, order=[]):
        obj_inds = []
        for t in trial_data:
            if len(np.unique(obj_inds)) == self.numObjs:
                break
            else:
                obj_inds.append(t[0])
        combs = list(itertools.combinations(np.unique(obj_inds), self.numResp))  
        CMs = [np.zeros((self.numResp, self.numResp)) for i in range(0, len(combs))]
        for trial in trial_data:
            distractor = [m for m in trial[2] if m != trial[0]]
            target = trial[0]
            pick = trial[1]
            possCombs = [[comb, idx] for idx, comb in enumerate(combs) if target in comb]
            for comb in possCombs:
                if set(distractor).issubset(set(comb[0])):
                    if len(order) > 0:
                        comb[0] = order
                    if pick == target:
                        idx = comb[0].index(pick)
                        CMs[comb[1]][idx, idx] += 1
                    elif pick != target:
                        CMs[comb[1]][comb[0].index(pick), comb[0].index(target)] += 1
                    else:
                        print('Matrix Error')
        return CMs, combs
               
    def computeSplitHalf(self, numSplits, subsample, verbose = False, correct = True, plot_ = False, subsel = None):
        import scipy.stats
        trial_data = self.trial_data
        Rs = []
        for s in range(0, numSplits):
            if verbose == True:
                print(s)
            else:
                pass
            np.random.shuffle(trial_data)
            half1 = trial_data[0:subsample/2]
            half2 = trial_data[-subsample/2:]
            if self.numResp == 2:
                CM1, combs = self.getPopCM2x2fast(half1, subsel = subsel)
                CM2, combs = self.getPopCM2x2fast(half2, subsel = subsel)
            else:
                CM1, combs = self.getPopCM(half1)
                CM2, combs = self.getPopCM(half2)
            half1_array = []
            half2_array = []
            for mat in range(0, len(CM1)):
                half1_array += list(normalize_CM(CM1[mat])[np.eye(CM1[mat].shape[0])==0])
                half2_array += list(normalize_CM(CM2[mat])[np.eye(CM2[mat].shape[0])==0])
            if self.corr_type == 'pearson':
                Rs.append(scipy.stats.pearsonr(half1_array, half2_array)[0])
                #correct = False
            else:
                Rs.append(scipy.stats.spearmanr(half1_array, half2_array)[0])
        if plot_ == True:
            plt.plot(half1_array, half2_array, 'b.')
        if correct == False:
            return Rs
        else:
            Rs_c = [SBcorrection(r, 2) for r in Rs]
            return Rs_c
      
    def imputeNtoM(self, use_objects):
        #Produces a single imputed matrix of a given size for given objects. The matrix will have blank entries
        #if you ask for a greater size than is given by the number of objects represented by your data
        obj_inds = []
        for t in self.trial_data:
            if len(np.unique(obj_inds)) == self.numObjs:
                break
            else:
                obj_inds.append(t[0])
        t = []
        for obj in use_objects:
            t.append(self.models.index(obj))
        import itertools
        combs = list(itertools.combinations(t, self.numResp))
        CM_imputed = np.zeros((len(t),len(t)))
        for trial in self.trial_data:
            for comb in combs:
                if set(comb).issubset(set(trial[2])):
                    if trial[0] == trial[1]:
                        CM_imputed[t.index(trial[0]), t.index(trial[0])] += 1
                    else:
                        CM_imputed[t.index(trial[1]), t.index(trial[0])] += 1
        return CM_imputed
    
class expDataDB_imageNet16(object):
    
    def __init__(self, collection, selector, numObjs):
        
        self.models = ['horse', 'boat', 'chair', 'plane', 'dog', 'pineapple',
                       'shirt', 'elephant', 'cat', 'car', 'truck', 'pumpkin',
                       'burger', 'book', 'turtle', 'bird']
        
        self.models_RV = ['lo_poly_animal_HRS_ARBN', 'MB29346', 'MB29822', 'MB31188', 'weimaraner', 
                          'single_pineapple', 'Colored_shirt_03M', 'lo_poly_animal_ELE_AS1', 
                          'foreign_cat', 'MB31620', 'MB29874', 'pumpkin_3', 'fast_food_23_1', 
                          'interior_details_033_2', 'lo_poly_animal_TRTL_B', 'lo_poly_animal_CHICKDEE']

        models_idxs = {}
        for idx, model in enumerate(self.models):
            models_idxs[model] = idx
        self.models_idxs = models_idxs
        
        models_RV_idxs = {}
        for idx, model in enumerate(self.models_RV):
            models_RV_idxs[model] = idx
        self.models_RV_idxs = models_RV_idxs
        
        conn = pymongo.Connection(port = 22334, host = 'localhost')
        db = conn.mturk
        col = db[collection]
        
        self.subj_data = col.find(selector, {'ImgData':1, 'Response':1, 'StimShown':1})
        self.trial_data = self.preprocess(self.subj_data)
        self.numResp = len(self.trial_data[0][2])
        self.numObjs = numObjs
        self.totalTrials = len(self.trial_data)
        self.corr_type = 'pearson'
        
    def init_from_pickle(self, pkFile):
        f = open(pkFile, 'rb')
        data = pk.load(f)
        f.close()
        self.subj_data = data
        self.trial_data = self.preprocess(self.subj_data)
        self.totalTrials = len(self.trial_data)
        
    def setPopCM(self):
        if self.numResp == 2:
            self.popCM, self.CM_order = self.getPopCM2x2fast(self.trial_data)
        else:
            self.popCM, self.CM_order = self.getPopCM(self.trial_data)
    
    def preprocess(self, subj_data):
        RV = [] #Response vector
        SV = [] #Stimulus vector
        DV = [] #Distractor vector
        for subj in subj_data:
            for r in subj['Response']:
                RV.append(r.split('/')[-1].split('.')[0][:-6])
            for s in subj['StimShown']:
                DV.append( [ s_.split('/')[-1].split('.')[0][:-6] for s_ in s if s_ != s[0] ] )
            for i in subj['ImgData']:
                SV.append(str(i[0]['obj']))
        new_data = []
        for idx, shown in enumerate(SV):
            model = shown
            CF_col_idx = self.models_idxs[model]
            CF_row_idx = self.models_RV_idxs[RV[idx]]
            new_data.append([CF_col_idx, CF_row_idx, [self.models_RV_idxs[m] for m in DV[idx]]]) #order is shown, picked, distractors
        return new_data
    
    def getPopCM2x2fast(self, trial_data, subsel = None):
        combs = list(itertools.combinations(range(0, self.numObjs), 2))
        CMs = {}
        for c in combs:
            CMs[c] = np.zeros((2,2))
        for t in trial_data:
            target = t[0]
            pick = t[1]
            cm = tuple(sorted(t[2])) #Because itertools always spits out the combs in sorted order
            if target == cm[0]:
                if target == pick:
                    CMs[cm][0,0] += 1
                else:
                    CMs[cm][1,0] += 1
            else:
                if target == pick:
                    CMs[cm][1,1] += 1
                else:
                    CMs[cm][0,1] += 1
        if subsel is None:
            return [CMs[c] for c in combs], combs
        else:
            return [CMs[c] for c in subsel], subsel
    
    def getPopCM(self, trial_data, order=[]):
        obj_inds = []
        for t in trial_data:
            if len(np.unique(obj_inds)) == self.numObjs:
                break
            else:
                obj_inds.append(t[0])
        combs = list(itertools.combinations(np.unique(obj_inds), self.numResp))  
        CMs = [np.zeros((self.numResp, self.numResp)) for i in range(0, len(combs))]
        for trial in trial_data:
            distractor = [m for m in trial[2] if m != trial[0]]
            target = trial[0]
            pick = trial[1]
            possCombs = [[comb, idx] for idx, comb in enumerate(combs) if target in comb]
            for comb in possCombs:
                if set(distractor).issubset(set(comb[0])):
                    if len(order) > 0:
                        comb[0] = order
                    if pick == target:
                        idx = comb[0].index(pick)
                        CMs[comb[1]][idx, idx] += 1
                    elif pick != target:
                        CMs[comb[1]][comb[0].index(pick), comb[0].index(target)] += 1
                    else:
                        print('Matrix Error')
        return CMs, combs
               
    def computeSplitHalf(self, numSplits, subsample, verbose = False, correct = True, plot_ = False, subsel = None):
        import scipy.stats
        trial_data = self.trial_data
        Rs = []
        for s in range(0, numSplits):
            if verbose == True:
                print(s)
            else:
                pass
            np.random.shuffle(trial_data)
            half1 = trial_data[0:subsample/2]
            half2 = trial_data[-subsample/2:]
            if self.numResp == 2:
                CM1, combs = self.getPopCM2x2fast(half1, subsel = subsel)
                CM2, combs = self.getPopCM2x2fast(half2, subsel = subsel)
            else:
                CM1, combs = self.getPopCM(half1)
                CM2, combs = self.getPopCM(half2)
            half1_array = []
            half2_array = []
            for mat in range(0, len(CM1)):
                half1_array += list(normalize_CM(CM1[mat])[np.eye(CM1[mat].shape[0])==0])
                half2_array += list(normalize_CM(CM2[mat])[np.eye(CM2[mat].shape[0])==0])
            if self.corr_type == 'pearson':
                Rs.append(scipy.stats.pearsonr(half1_array, half2_array)[0])
                #correct = False
            else:
                Rs.append(scipy.stats.spearmanr(half1_array, half2_array)[0])
        if plot_ == True:
            plt.plot(half1_array, half2_array, 'b.')
        if correct == False:
            return Rs
        else:
            Rs_c = [SBcorrection(r, 2) for r in Rs]
            return Rs_c
      
    def imputeNtoM(self, use_objects):
        #Produces a single imputed matrix of a given size for given objects. The matrix will have blank entries
        #if you ask for a greater size than is given by the number of objects represented by your data
        obj_inds = []
        for t in self.trial_data:
            if len(np.unique(obj_inds)) == self.numObjs:
                break
            else:
                obj_inds.append(t[0])
        t = []
        for obj in use_objects:
            t.append(self.models.index(obj))
        import itertools
        combs = list(itertools.combinations(t, self.numResp))
        CM_imputed = np.zeros((len(t),len(t)))
        for trial in self.trial_data:
            for comb in combs:
                if set(comb).issubset(set(trial[2])):
                    if trial[0] == trial[1]:
                        CM_imputed[t.index(trial[0]), t.index(trial[0])] += 1
                    else:
                        CM_imputed[t.index(trial[1]), t.index(trial[0])] += 1
        return CM_imputed