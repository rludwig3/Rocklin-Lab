import numpy as np
import pandas as pd
import sys
import copy
from collections import Counter
import time
import pymzml
import pickle
import _pickle as cpickle
import scipy as sp
import tensorly
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import matplotlib as mpl
import math
import re
import zlib
import statistics
import importlib.util
import LCMSTensorAnalysis as TA 
from scipy.optimize import curve_fit
from scipy.fftpack import fft
from tensorly.decomposition import non_negative_parafac
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelmax
from mpl_toolkits.mplot3d import Axes3D
pd.options.display.max_columns = None
sns.set_context('notebook')
from IPython.display import display

skyinfo=pd.read_csv('/projects/p30802/wes/hdx_analysis/180604_full_fixts2.csv')

hd_mass_diff = 1.006277
c13_mass_diff = 1.00335
total_isotopes = 60
radius = 30

est_peak_gaps = [0] + list(np.linspace(c13_mass_diff,hd_mass_diff,7)) + [hd_mass_diff for x in range(total_isotopes - 8)]
cum_peak_gaps = np.cumsum(est_peak_gaps)

mz_lows = []
mz_highs = []
mz_centers = []
low_lims = []
high_lims = []

count = 0
for i in range(len(skyinfo)):
    mz_centers.append(skyinfo['obs_mz'].values[i] + (cum_peak_gaps / skyinfo['charge'].values[i]))
    mz_lows.append(skyinfo['obs_mz'].values[i] - (10.0/skyinfo['charge'].values[i]))
    mz_highs.append(skyinfo['obs_mz'].values[i] + (60.0/skyinfo['charge'].values[i]))

    low_lims.append(mz_centers[count] * ((1000000.0 - radius)/1000000.0))
    high_lims.append(mz_centers[count] * ((1000000.0 + radius)/1000000.0))
    count += 1

mz_low = min(mz_lows)
mz_high = max(mz_highs)

protein_names = dict.fromkeys(skyinfo['name'].values)
for key in protein_names.keys():
    protein_names[key]=skyinfo.loc[skyinfo['name'] == key]

timepoints = ["UN_", "UN2_", "UN3_", "10s_", "16s_", "26s_", "40s_", "65s_", "105s_", 
              "160s_", "4m30_", "7m10_", "11m30s_", "17m30s_", "28m_", "44m_", "70m_", 
              "1hr50_", "3hr10_", "5hr15_", "9hr20_", "17hr45_", "28hr_"]

name_list = list(protein_names.keys())

path = '/projects/p30802/wes/hdx_analysis/stg1/'
name = sys.argv[1]
name_files = glob.glob1(path, "*"+name+"*")
tp_files = []
tp_cluster_files =[]
for tp, n_mzml in zip(timepoints, range(len(timepoints))):

    """
    construct map of RT clusters in dict of protein names, containing dicts of charge state skyinfo lines and
    other skyinfo lines in cluster. 
    """

    protein_rts = dict.fromkeys(skyinfo['name'].values)
    rt_matches = dict.fromkeys(skyinfo['name'].values)
    for key in protein_names.keys():
        protein_rts[key]=protein_names[key][["Unnamed: 0","RT_"+str(n_mzml)]].values
        rt_matches[key] = dict.fromkeys(protein_rts[key][:,0])
        for tup in protein_rts[key]:
            #np array allows for tuple slicing of list (no longer used here)
            rt_matches[key][tup[0]] = np.array([x for x in protein_rts[key] if abs(x[1]-tup[1])<=0.2])

    #Creates set of the RT clusters for each tp of a protein, w/o duplicates to be iterated over        
    rt_tp_clusters = set()
    for n_line in rt_matches[name].keys():
        if set(rt_matches[name][n_line][:,0].flatten()).issubset(rt_tp_clusters):
            pass
        else:
            rt_tp_clusters.add(tuple(rt_matches[name][n_line][:,0]))        
    r = re.compile(tp)
    tp_files.append(list(filter(r.search, name_files)))
    rt_cluster_files = []
    for cluster in rt_tp_clusters: 
        rtcf_buffer = []
        for line in cluster:
            r = re.compile(str(int(line)))
            rtcf_buffer.append(list(filter(r.search, tp_files[n_mzml]))[0])
        rt_cluster_files.append(rtcf_buffer)
    tp_cluster_files.append(rt_cluster_files)

n_factors = 10
all_tp_clusters = []
all_factors = []
count = 0
for tp in tp_cluster_files:
    print("TP: "+timepoints[count])
    tp_clusters = []
    tp_cluster_info = []
    tp_factors = []
    for rt_cluster in tp:
        DataTensors = []
        concat_ins = []
        concat_charges = []
        concat_DTs = []
        elapsed = 0
        for file in rt_cluster:
            sky_idx = int(file.split("_")[0])
            #Create individual DataTensors by zlib.decompress() and cpickle.loads()
            output = cpickle.loads(zlib.decompress(open(path+file, 'rb').read()))
            newDataTensor = TA.DataTensor(sky_idx, charge_state = (sky_idx, skyinfo["charge"].values[sky_idx], len(output[1])), rts = output[0], dts= output[1], seq_out=output[2], int_seq_out=None)
            newDataTensor.lows = np.searchsorted(newDataTensor.mz_bins, low_lims[sky_idx])
            newDataTensor.highs = np.searchsorted(newDataTensor.mz_bins, high_lims[sky_idx])
            newDataTensor.decomposition_series(n_factors, n_factors+1, gauss_params=(3,1))
            DataTensors.append(newDataTensor)
            
        #Check output clusters to determine which tensors if any should be combined, start by acquiring list of
        #all isotopic clusters from all decompositions
        #full_cluster_info=[]
        for i in DataTensors:
            for j in i.decomps:
                for k in j:
                    tp_factors.append(k)
                    for l in k.isotope_clusters:
                        tp_clusters.append(l)
                        tp_cluster_info.append(l.info_tuple) 

        protein_clusters=pd.DataFrame(tp_cluster_info)
        protein_clusters.columns=("TensorIDX", "NFactors", "FactorIDX", "ClusterIDX", "LowMZIDX", "HighMZIDX", "AUC", "GrateSum", 
                    "GrateAreaRatio","BaselineAUC", "BaselineGrateSum", "BaselineGAR", "PeakError", "BaselinePeakError", 
                    "CompositeScore", "RTs", "DTs", "MaxRTDT", "OuterRTDT", "NormFits", "AbsCOM","IntMZ")

        top_clusters = protein_clusters.sort_values(by = ['BaselineGAR', 'BaselinePeakError'], ascending = False)[:math.ceil(len(protein_clusters)/5)]

        #cheks that each tensor gives 85% of the even split contribution to the top 20% of clusters, if less the tensor is not included in the concatenated data
        #85% is an arbitrary choice of cutoff value, empirically determine point at which adding DT to combined harms quality
        if len(output)>0:    
            for i in range(len(DataTensors)):
                contribution = len(top_clusters.loc[top_clusters['TensorIDX']==i])
                if contribution > (math.floor(math.ceil(len(protein_clusters)/5)/len(DataTensors))*0.85):
                    concat_ins.append(DataTensors[i].interpolate(DataTensors[i].full_grid_out, 5000, (3,1))[0])
                    concat_charges.append(DataTensors[i].charge_state)
            if len(concat_ins)>1:
                interpolated_lows, interpolated_highs = DataTensors[0].interpolate(DataTensors[0].full_grid_out, new_mz_len=5000, gauss_params=(3,1))[1:3]
                concat_DTs += output[i][1]
                concatenated=np.concatenate(concat_ins, axis=1)
                concat_tuples = np.asarray([ic.charge_state for ic in concat_ins])
                concat_charges =[]
                for tup in concat_tuples:
                    concat_charges.append(tup)
                concat_charges = np.asarray(concat_charges)
                DataTensors.append(TA.DataTensor(len(DataTensors)+1, charge_state = concat_charges, rts = output[i][0], dts = np.array(concat_DTs), n_concatenated=len(concat_ins), concatenated_grid=concatenated, 
                                              lows = interpolated_lows, highs = interpolated_highs))
                DataTensors[-1].decomposition_series(n_factors, n_factors+1)   
        
                #Add concatenated
                for j in DataTensors[-1].decomps:
                    for k in j:
                        tp_factors.append(k)

    all_factors.append(tp_factors)
    count += 1
    
with open(name+"_"+str(n_factors)+"_factors"+".cpickle.zlib", 'wb') as file:
    file.write(zlib.compress(cpickle.dumps(all_factors)))

all_tp_clusters = []
for tp in all_factors:
    buffer = []
    for factor in tp:
        for ic in factor.isotope_clusters:
            buffer.append(ic)
    all_tp_clusters.append(buffer)

#BEGIN PATH OPTIMIZER
#PREFILTER ICs
def max_mz(ic):
    return int(ic.baseline_integrated_mz.index(max(ic.baseline_integrated_mz)))

#prefiltered_ics has same shape as all_tp_clusters, but collapses 3 UN timepoints into 1
prefiltered_ics=[]

#pre_scoring has same order as a single tp of all_tp_clusters
pre_scoring=[]

#idx_scores contains the sum of the weighted BPE/BGAR sorted indices of ics in prefiltered_ics, with pre_scoring order
idx_scores = []

#Prefilitering UN clusters by integration box alignment to low mz
prefiltered_ics.append([])
for tp in all_tp_clusters[:3]:
    for ic in tp:
        if ic.baseline_integrated_mz.index(max(ic.baseline_integrated_mz))<5:
            prefiltered_ics[0].append(ic)
            
#Further filter each tp by BGAR and PeakError

#Decorate
for ic in prefiltered_ics[0]:
    pre_scoring.append([ic.auc, ic.baseline_grate_sum/ic.baseline_auc, ic.baseline_peak_error])
    
#Sort, add weights here for tuning
AUC_sorted = sorted(pre_scoring, key = lambda x: x[0], reverse = True)
BGAR_sorted = sorted(pre_scoring, key = lambda x: x[1], reverse = True)
BPE_sorted = sorted(pre_scoring, key = lambda x: x[2])

#Undecorate
for tup in pre_scoring:
    idx_scores.append(sum([AUC_sorted.index(tup), BGAR_sorted.index(tup), BPE_sorted.index(tup)]))
idx_scores_sorted=sorted(idx_scores, key=lambda x: x)
score_filter=idx_scores_sorted[math.ceil(len(idx_scores_sorted)/5)]

buffer=[]
for i in range(len(idx_scores)):
    if idx_scores[i]<=score_filter:
        buffer.append(prefiltered_ics[0][i])

#Reset prefiltered UN tp to BPE and BGAR filtered ics
prefiltered_ics[0]=buffer

#Repeat process for remaining timepoints
for tp in range(4, len(all_tp_clusters)):
    #Reset tp vars
    pre_scoring=[]
    idx_scores=[]
    buffer=[]
    
    #Decorate
    for ic in all_tp_clusters[tp]:
        pre_scoring.append([ic.baseline_grate_sum/ic.baseline_auc, ic.baseline_peak_error])
    
    #Sort, add weights for tuning
    AUC_sorted = sorted(pre_scoring, key = lambda x: x[0], reverse = True)
    BGAR_sorted=sorted(pre_scoring, key=lambda x: x[0], reverse=True)
    BPE_sorted=sorted(pre_scoring, key=lambda x: x[1])
    
    #Undecorate
    for tup in pre_scoring:
        idx_scores.append(sum([AUC_sorted.index(tup), BGAR_sorted.index(tup), BPE_sorted.index(tup)]))
    idx_scores_sorted=sorted(idx_scores, key=lambda x: x)
    score_filter=idx_scores_sorted[math.ceil(len(idx_scores_sorted)/5)]
    
    buffer=[]
    #CONSIDER SORTING BUFFER BY SCORE VALUES TO APPLY LOW IDX PREFERENTIAL DISTRIBUTION FOR SAMPLING
    for i in range(len(idx_scores)):
        if idx_scores[i]<=score_filter:
            buffer.append(all_tp_clusters[tp][i])
            
    prefiltered_ics.append(buffer)   

prefiltered_ic_mzs = []
for tp in prefiltered_ics:
    tp_buffer = []
    for ic in tp:
        tp_buffer.append(int(ic.baseline_integrated_mz.index(max(ic.baseline_integrated_mz))))
    prefiltered_ic_mzs.append(tp_buffer)
    

sample_paths = []
n_fails = 0
while len(sample_paths)<200:
    if n_fails < 1000:
        cur_path = []
        prev = np.random.randint(0,len(prefiltered_ics[0]), 1, dtype=int)[0]
        cur_path.append(prefiltered_ics[0][prev])
        for tp in range(1, 20):
            #print("TP: "+str(tp)+" Prev: "+str(prev)+" Max_MZ: "+str(max_mz(prefiltered_ics[tp-1][prev])))
            #generate list of ics with "rightward" mzs
            sublist = [prefiltered_ics[tp].index(x) for x in prefiltered_ics[tp] if max_mz(x) - max_mz(prefiltered_ics[tp-1][prev]) > -2]
            if len(sublist)>0:
                prev = sublist[np.random.randint(0,len(sublist), 1, dtype=int)[0]]
                cur_path.append(prefiltered_ics[tp][prev])
                if len(cur_path) == 20:
                    sample_paths.append(cur_path)
            else:
                n_fails += 1
                break
                
    else:
        cur_path = []
        prev = np.random.randint(0,len(prefiltered_ics[0]), 1, dtype=int)[0]
        cur_path.append(prefiltered_ics[0][prev])
        for tp in range(1, 20):
            prev = np.random.randint(0,len(prefiltered_ics[tp]), 1, dtype=int)[0]
            cur_path.append(prefiltered_ics[tp][prev])
            if len(cur_path) == 20:
                    sample_paths.append(cur_path)

#DEFINE SCORING FXNS

#Root Mean Squared Error
def rt_rmse(ics): 
    rt_centers = []
    for ic in ics:
        rt_centers.append(ic.RT_center)
    sample_mean = statistics.mean(rt_centers)
    mse = 0
    for rt in rt_centers:
        mse += (rt-sample_mean)**2
    return math.sqrt(mse)

def dt_fit_sum(ics):
    rsum = 0
    #Compare like charges by forcing length matches when reasonable (difference less than 3?)
    #all to all comparison check? -> unequal number of comparisons
    #Compare like to like? -> prefers non-like DTs if no penalty for non-likeness
    #sort into charge states, compare single charge states to eachother and break apart concat charges to compare individually to like-charges (forcing length matching when needed)
    #charges is dict of dicts: charges = { 3.0: {dt_lengths: [ics]}, 4.0: {dt_lengths: [ics]}, 5.0: {dt_lengths: [ics]}, "concat": [ics] }
    #each ic gets a single comparison, unless there is no charge state and length match, consider penalty (0.1 or something similar)
    charges = {}
    charges["concat"] = []
    
    #SORT
    for ic in ics:
        if type(ic.charge_state) == tuple:
            if ic.charge_state[1] in charges.keys():
                if ic.charge_state[2] in charges[ic.charge_state[1]].keys():
                    charges[ic.charge_state[1]][ic.charge_state[2]].append(ic)
                else:
                    charges[ic.charge_state[1]][ic.charge_state[2]] = []
                    charges[ic.charge_state[1]][ic.charge_state[2]].append(ic)
            else:
                charges[ic.charge_state[1]] = {}
                charges[ic.charge_state[1]][ic.charge_state[2]] = []
                charges[ic.charge_state[1]][ic.charge_state[2]].append(ic)
        else:
            charges["concat"].append(ic)

    #SCORE
    for key1 in charges.keys():        
        if key1 == "concat":
            for ic in charges[key1]:
                #cut apart dts by ic.charge_state information
                #look into charges to find matching charge state and dt length
                #compare each single dt to a match, store results, rsum += value closest to 1
                #ic.charge_state is an np_array of n single-charge charge_state tuples. Can use advanced indexing

                single_dts = []
                #generate list of single dts
                #charge_state tuples are in order of concatenation, 0 is leftmost moving to right
                for i in range(np.shape(ic.charge_state)[0]):
                    #set low idx to sum of lengths of preceeding dts
                    single_dts.append(ic.dts[int(sum(ic.charge_state[:i, 2])):int(sum(ic.charge_state[:i, 2])+ic.charge_state[i, 2])])

                #for each single dt in concat_dts, match the charge state and length
                concat_scores = []
                for i in range(np.shape(ic.charge_state)[0]):
                    if ic.charge_state[i,2] in charges[ic.charge_state[i,1]].keys():
                        concat_scores.append(np.dot(single_dts[i]/np.linalg.norm(single_dts[i]), charges[ic.charge_state[i,1]][ic.charge_state[i,2]][0].dts/np.linalg.norm(charges[ic.charge_state[i,1]][ic.charge_state[i,2]][0].dts)))
                    else:
                        for alt_key in charges[ic.charge_state[i,1]].keys():
                            if abs(ic.charge_state[i,2]-alt_key) < 3 and ic.charge_state[i,2] != alt_key:
                                if ic.charge_state[i,2]-alt_key > 0:
                                    concat_scores.append(np.dot(single_dts[i][:len(charges[ic.charge_state[i,1]][alt_key].dts)]/np.linalg.norm(single_dts[i][:len(charges[ic.charge_state[i,1]][alt_key].dts)]), 
                                                                charges[ic.charge_state[i,1]][alt_key].dts/np.linalg.norm(charges[ic.charge_state[i,1]][alt_key].dts)))
                                else:
                                    concat_scores.append(np.dot(single_dts[i]/np.linalg.norm(single_dts[i]), 
                                                                charges[ic.charge_state[i,1]][alt_key].dts[:len(single_dts[i])]/np.linalg.norm(charges[ic.charge_state[i,1]][alt_key].dts[:len(single_dts[i])])))
                            else:
                                concat_scores.append(0.1)
                #take best match of single dt in concat_ic.dts              
                rsum += max(concat_scores)

        else:     
            for key2 in charges[key1].keys():
                if len(charges[key1][key2]) > 1:
                    for i in range(len(charges[key1][key2])-1):
                        rsum += np.dot(charges[key1][key2][i].dts/np.linalg.norm(charges[key1][key2][i].dts), charges[key1][key2][i+1].dts/np.linalg.norm(charges[key1][key2][i+1].dts))
                else:
                    for alt_key in charges[key1].keys():
                        if abs(key2-alt_key) < 3 and key2 != alt_key:
                            if key2-alt_key > 0:
                                rsum += np.dot(charges[key1][key2][0].dts[:len(charges[key1][alt_key][0].dts)]/np.linalg.norm(charges[key1][key2][0].dts[:len(charges[key1][alt_key][0].dts)]), charges[key1][alt_key][0].dts/np.linalg.norm(charges[key1][alt_key][0].dts)) 
                            else:
                                rsum += np.dot(charges[key1][key2][0].dts/np.linalg.norm(charges[key1][key2][0].dts), charges[key1][alt_key][0].dts[:len(charges[key1][key2][0].dts)]/np.linalg.norm(charges[key1][alt_key][0].dts[:len(charges[key1][key2][0].dts)])) 
                        else:
                            rsum += 0.1
    
    #return inverse of avg fit, penalizes bad fits with higher score
    return len(ics)/rsum

def rt_fit_sum(ics):
    rsum = 0 
    lens = {}
    
    for ic in ics: 
        if len(ic.rts) in lens.keys():
            lens[len(ic.rts)].append(ic)
        else:
            lens[len(ic.rts)] = []
            lens[len(ic.rts)].append(ic)
    
    for key in lens.keys():
        if len(lens[key]) > 1:
            for i in range(len(lens[key])-1):
                rsum += np.dot(lens[key][i].rts/np.linalg.norm(lens[key][i].rts), lens[key][i+1].rts/np.linalg.norm(lens[key][i+1].rts))
                
    return rsum

def bpe(ics): 
    bpes=[]
    for ic in ics:
        bpes.append(ic.baseline_peak_error)
    return sum(bpes)

#Maximize
def bgar(ics): 
    BGARs=[]
    for ic in ics:
        BGARs.append(ic.baseline_grate_sum/ic.baseline_auc) 
    return sum(BGARs)
    
#Minimize difference between max int_mz intensities
def max_mzs(ics): 
    max_mzs=[]
    for ic in ics:
        max_mzs.append(max(ic.baseline_integrated_mz))
    mean_max_mz = statistics.mean(max_mzs)
    return sum([abs(x-mean_max_mz) for x in max_mzs])
        
#Minimize Residual to fitted line for later ic selection, initially minimize by fit error
def int_com(ics): 
    coms=[]
    for ic in ics:
        coms.append(ic.baseline_integrated_mz.index(max(ic.baseline_integrated_mz)))
    return np.polynomial.polynomial.Polynomial.fit(range(len(coms)), coms, 0, full=True)[1][0][0]

def auc(ics):
    rsum = 0
    for ic in ics:
        rsum += ic.baseline_auc
    return rsum

def mz_alignment(ics):
    
    def mz_center(ic):
        return ic.baseline_integrated_mz.index(max(ic.baseline_integrated_mz))
    
    rsum = 0
    
    for i in range(1, len(ics)-1):
        d1 = mz_center(ics[i+1])-mz_center(ics[i])
        d2 = mz_center(ics[i])-mz_center(ics[i-1])
        if d1 < -2:
            rsum += d1**4
        if d2 < -2:
            rsum += d2**4
    
    return rsum
        
        
rt_weight, dt_weight, bpe_weight, bgar_weight, max_mz_weight, int_com_weight, auc_weight, mz_align_weight = 1000, 100000, 1000, 10000, 0.00001, 0.00001, 5000000000, 10

def combo_score(ics):
    #return sum([rt_weight*rt_fit_sum(ics), bpe_weight*bpe(ics), bgar_weight*1/bgar(ics), max_mz_weight*max_mzs(ics), int_com_weight*int_com(ics), auc_weight*1/auc(ics)])
    return sum([rt_weight*rt_fit_sum(ics), dt_weight*dt_fit_sum(ics), auc_weight*1/auc(ics), mz_align_weight*mz_alignment(ics)])

#ITERATIVELY MINIMIZE PATHS

final_paths = []
for sample in sample_paths:
    current = copy.deepcopy(sample)
    edited = True
    while edited:
        
        edited = False
        n_changes = 0
        ic_indices=[]
        alt_paths=[]
        
        for tp in range(len(current)):
            for ic in prefiltered_ics[tp]:
                buffer=copy.copy(current)
                buffer[tp]=ic
                alt_paths.append(buffer)

        #Decorate alt_paths
        combo_scoring = []
        for pth in alt_paths:
            combo_scoring.append(combo_score(pth))

        if min(combo_scoring) < combo_score(current):
            current = alt_paths[combo_scoring.index(min(combo_scoring))]
            n_changes+=1
            edited=True

        current_score = combo_score(current) 

        if edited == False:
            final_paths.append(current)

final_scores = []
for pth in final_paths:
    final_scores.append(combo_score(pth))
winner = final_paths[final_scores.index(min(final_scores))]

runners = []
for tp in prefiltered_ics:
    runners.append(sorted(tp, key = lambda x: x.auc)[:5])

with open(name+"_"+str(n_factors)+"_winning_path.cpickle.zlib", "wb") as file:
    file.write(zlib.compress(cpickle.dumps(winner)))

with open(name+"_"+str(n_factors)+"runners.cpickle.zlib", 'wb') as file:
    file.write(zlib.compress(cpickle.dumps(runners)))







