import numpy as np
import pandas as pd
import sys

import os
import psutil

import copy
from collections import Counter
import time
import pymzml
import _pickle as cpickle
import zlib

n_mzml = int(sys.argv[1])
skyinfo=pd.read_csv('../180604_full_fixts2.csv')
skyinfo['n'] = range(len(skyinfo))

skyinfo['Drift Time MS1'] = skyinfo['im_mono'] / 200.0 * 13.781163434903


#This is a redundant implementation of RT_Cluster identification, this could likely be performed with only the RT_matches dict and skyinfo
#Use in final output to select keep_scan_times high and low boundaries. 
protein_names = dict.fromkeys(skyinfo['name'].values)
protein_RTs = dict.fromkeys(skyinfo['name'].values)
RT_matches = dict.fromkeys(skyinfo['name'].values)
for key in protein_names.keys():
    protein_names[key]=skyinfo.loc[skyinfo['name']==key]
    protein_RTs[key]=protein_names[key][["Unnamed: 0","RT_"+str(n_mzml)]].values
    RT_matches[key] = dict.fromkeys(protein_RTs[key][:,0])
    for tup in protein_RTs[key]:
        RT_cluster = np.array([x for x in protein_RTs[key] if abs(x[1]-tup[1])<=0.2])
        lo_line = [x[0] for x in RT_cluster if x[1] == min(RT_cluster[:,1])]
        hi_line = [x[0] for x in RT_cluster if x[1] == max(RT_cluster[:,1])]
        RT_matches[key][tup[0]]=(lo_line + hi_line)


mzmls="""180604_Mix2_MES_nonlin_UN.mzML
180604_Mix2_MES_nonlin_UN2.mzML
180604_Mix2_MES_nonlin_UN3.mzML
180604_Mix2_MES_nonlin_10s.mzML
180604_Mix2_MES_nonlin_16s.mzML
180604_Mix2_MES_nonlin_26s.mzML
180604_Mix2_MES_nonlin_40s.mzML
180604_Mix2_MES_nonlin_65s.mzML
180604_Mix2_MES_nonlin_105s.mzML
180604_Mix2_MES_nonlin_160s.mzML
180604_Mix2_MES_nonlin_4m30.mzML
180604_Mix2_MES_nonlin_7m10.mzML
180604_Mix2_MES_nonlin_11m30s.mzML
180604_Mix2_MES_nonlin_17m30s.mzML
180604_Mix2_MES_nonlin_28m.mzML
180604_Mix2_MES_nonlin_44m.mzML
180604_Mix2_MES_nonlin_70m.mzML
180604_Mix2_MES_nonlin_1hr50.mzML
180604_Mix2_MES_nonlin_3hr10.mzML
180604_Mix2_MES_nonlin_5hr15.mzML
180604_Mix2_MES_nonlin_9hr20.mzML
180604_Mix2_MES_nonlin_17hr45.mzML
180604_Mix2_MES_nonlin_28hr.mzML""".split('\n')

mzml = mzmls[n_mzml]
mzml_in = '../' + mzml

ret_ubounds=skyinfo['RT_%s' % n_mzml].values+0.4
ret_lbounds=skyinfo['RT_%s' % n_mzml].values-0.4

dt_ubounds=skyinfo['Drift Time MS1'].values * 1.06
dt_lbounds=skyinfo['Drift Time MS1'].values * 0.94
output=[]

drift_times=[]
scan_times = []

with open(mzml_in) as file:
    lines = file.readlines()
for line in lines:
    if '<cvParam cvRef="MS" accession="MS:1002476" name="ion mobility drift time" value' in line:
        dt=line.split('value="')[1].split('"')[0]#replace('"/>',''))
        drift_times.append(float(dt))

for line in lines:
    if '<cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value=' in line:
        st=line.split('value="')[1].split('"')[0]#replace('"/>',''))
        scan_times.append(float(st))

drift_times = np.array(drift_times)
scan_times = np.array(scan_times)
scan_numbers=np.arange(0,len(scan_times))

process = psutil.Process(os.getpid())
print (process.memory_info().rss)
msrun = pymzml.run.Reader(mzml_in)
print (process.memory_info().rss)
k = 0



starttime = time.time()

print (time.time() - starttime, mzml)

hd_mass_diff=1.006277
c13_mass_diff=1.00335

total_isotopes = 50
radius = 30
est_peak_gaps=[0] + list(np.linspace(c13_mass_diff,hd_mass_diff,7)) + [hd_mass_diff for x in range(total_isotopes - 8)]
cum_peak_gaps = np.cumsum(est_peak_gaps)

with open('%s.proc4.progress' % mzml,'w') as file:
    file.write('%s start\n' % (time.time() - starttime))


scan_to_lines=[[] for i in scan_times]
scans_per_line = []
output_scans = [[] for i in range(len(skyinfo))]

for i in range(len(skyinfo)):
    #print i
    #sys.stdout.flush()
    name = skyinfo.iloc[i]['name']
    RT_lo = int(RT_matches[name][i][0])
    RT_hi = int(RT_matches[name][i][1])
    keep_scans = scan_numbers[(drift_times >= dt_lbounds[i]) & (drift_times <= dt_ubounds[i]) & (scan_times <= ret_ubounds[RT_hi]) & (scan_times >= ret_lbounds[RT_lo])]
    scans_per_line.append(len(keep_scans))
    for scan in keep_scans:
        scan_to_lines[scan].append(i)

    if i % 100 == 0: print(str(i)+" lines, time: "+str(time.time()-starttime))

for scan_number, scan in enumerate(msrun):#i in sorted(keep_scans)[0:1000]:

    if scan_number % 100 == 0: print (scan_number, process.memory_info().rss / (1024*1024*1024), (len(skyinfo) - output_scans.count([])) / len(skyinfo) )

    if len(scan_to_lines[scan_number]) > 0:
        spectrum = np.array(scan.peaks('reprofiled')).astype(np.float32)
        if len(spectrum) == 0: spectrum = scan.peaks('raw').astype(np.float32)
        spectrum=spectrum[spectrum[:,1] > 10]

    for i in scan_to_lines[scan_number]:
        mz_low = skyinfo['obs_mz'].values[i] - (10.0/skyinfo['charge'].values[i])
        mz_high = skyinfo['obs_mz'].values[i] + (60.0/skyinfo['charge'].values[i])
        try:
            output_scans[i].append(spectrum[(mz_low < spectrum[:,0]) & (spectrum[:,0] < mz_high)])
        except:
            print (i, output_scans[i], mz_low, mz_high)
            print (spectrum)
            print (spectrum[(mz_low < spectrum[:,0]) & (spectrum[:,0] < mz_high)])
            sys.exit(0)
        if len(output_scans[i]) == scans_per_line[i]:
            hdx_time = mzml[23:-5]
            name = skyinfo.iloc[i]['name']
            #RT_start = skyinfo.iloc[i]['RT_'+str(sys.argv[1])]
            RT_lo = int(RT_matches[name][i][0])
            RT_hi = int(RT_matches[name][i][1])
            keep_drift_times = drift_times[(drift_times >= dt_lbounds[i]) & (drift_times <= dt_ubounds[i]) & (scan_times <= ret_ubounds[i]) & (scan_times >= ret_lbounds[i])]
            keep_scan_times = scan_times[(drift_times >= dt_lbounds[i]) & (drift_times <= dt_ubounds[i]) & (scan_times <= ret_ubounds[RT_hi]) & (scan_times >= ret_lbounds[RT_lo])]
            output = [sorted(set(keep_scan_times)), sorted(set(keep_drift_times)), output_scans[i]]
            with open(str(i)+"_"+hdx_time+"_"+name+".cpickle.zlib", 'wb') as file:
                file.write(zlib.compress(cpickle.dumps(output)))
            print (scan_number, process.memory_info().rss / (1024*1024*1024), 'presave')
            output_scans[i] = []
            print (scan_number, process.memory_info().rss / (1024*1024*1024), 'savedisk')
            
    if len(scan_to_lines[scan_number]) > 0:
        cur_lengths = np.array([len(output_scans[i]) for i in scan_to_lines[scan_number]])
        target_lengths = np.array([scans_per_line[i] for i in scan_to_lines[scan_number]])
