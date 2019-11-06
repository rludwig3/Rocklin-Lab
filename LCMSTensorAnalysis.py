import numpy as np
import scipy as sp
import tensorly
import copy
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import peakutils
import glob
import time
import matplotlib as mpl
from tensorly.decomposition import non_negative_parafac
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter


#Definitions of Classes built from overlapping_charge_states.ipynb

#Holds the data for a single identified cluster, generated from subspace_selector.ipynb
class DataTensor:
    
    def __init__(
            self, tensor_idx, charge_state = None, rts = None, dts = None, seq_out = None, int_seq_out = None,
            n_concatenated = 1, concatenated_grid = None, lows = None, highs = None, abs_mz_low = None):
        
        self.tensor_idx = tensor_idx
        self.n_concatenated = n_concatenated
        self.charge_state = charge_state
        
        if self.n_concatenated == 1:
            self.rts = np.array(rts)
            self.dts = np.array(dts)
            self.seq_out = np.array(seq_out)
            if int_seq_out != None:
                self.int_seq_out = np.array(int_seq_out)
                self.int_seq_out_float = self.int_seq_out.astype('float64')
                self.int_grid_out = np.reshape(self.int_seq_out_float, (len(rts), len(dts), 50))
                self.int_gauss_grids = self.gauss(self.int_grid_out)

            self.grid_out = np.reshape(seq_out, (len(rts), len(dts)))

            #For creating full_grid_out, a 3d array with all mz dimensions having the same length
            #Find min and max mz range values:
            #check head and tail of each grid_out index ndarray and compare them
            self.mz_bin_high = 0
            self.mz_bin_low = 10000000
            for j in range(len(self.grid_out)):
                for k in range(len(self.grid_out[j])):
                    if np.shape(self.grid_out[j][k])[0] != 0:
                        if self.grid_out[j][k][0].item(0) < self.mz_bin_low:
                            self.mz_bin_low = self.grid_out[j][k][0].item(0)
                        if self.grid_out[j][k][-1].item(0) > self.mz_bin_high:
                            self.mz_bin_high = self.grid_out[j][k][-1].item(0)

            #create zero array with range of bin indices
            self.mz_bins = np.arange(self.mz_bin_low, self.mz_bin_high, 0.002)
            self.mz_len = len(self.mz_bins)

            #create empty space with dimensions matching grid_out and m/z indices
            self.full_grid_out=np.zeros((np.shape(self.grid_out)[0], np.shape(self.grid_out)[1], self.mz_len))

            #determine and apply mapping of nonzero grid_out values to full_grid_out
            for l in range(len(self.grid_out)):
                for m in range(len(self.grid_out[l])):
                    self.indices = np.clip(np.searchsorted(self.mz_bins, self.grid_out[l][m][:,0]), 0, len(self.mz_bins)-1)
                    self.full_grid_out[l,m][self.indices] = self.grid_out[l][m][:,1]
            self.full_grid_out = self.full_grid_out

            self.full_gauss_grids = self.gauss(self.full_grid_out)
            
        else:
            self.dts = dts
            self.lows = lows
            self.highs = highs
            self.concatenated_grid = concatenated_grid
            self.mz_bin_low = abs_mz_low
        
    #Takes tensor input and gaussian filter parameters, outputs filtered data
    def gauss(self, grid, rt_sig = 3, dt_sig = 1):
        gauss_grid=np.zeros(np.shape(grid))
        for i in range(np.shape(grid)[2]):
            gauss_grid[:,:,i]=gaussian_filter(grid[:,:,i],(rt_sig, dt_sig))
        return gauss_grid
    
    #Takes length of mz_bins to interpolated to, and optional gaussian filter parameters
    #Returns the interpolated tensor, length of interpolated axis, interpolated low_lims and high_lims
    def interpolate(self, grid_in, new_mz_len, gauss_params = None):
        
        if gauss_params!=None:
            grid=self.gauss(grid_in, gauss_params[0], gauss_params[1])
        else:
            grid=grid_in
            
        test_points = []
        z_axis = np.clip(np.linspace(0, np.shape(grid)[2], new_mz_len), 0, np.shape(grid)[2]-1)
        for n in range(np.shape(grid)[0]):
            for o in range(np.shape(grid)[1]):
                for p in z_axis:
                    test_points.append((n, o, p))
        x, y, z = np.arange(np.shape(grid)[0]), np.arange(np.shape(grid)[1]), np.arange(np.shape(grid)[2])
        interpolation_function=sp.interpolate.RegularGridInterpolator(points = [x, y, z], values = grid)
        interpolated_out=interpolation_function(test_points)
        interpolated_out=np.reshape(interpolated_out, (np.shape(grid)[0], np.shape(grid)[1], new_mz_len))
        
        interpolated_bin_mzs = np.linspace(self.mz_bin_low, self.mz_bin_high, new_mz_len)
        interpolated_low_lims = np.searchsorted(interpolated_bin_mzs, self.mz_bins[self.lows])
        interpolated_high_lims = np.searchsorted(interpolated_bin_mzs, self.mz_bins[self.highs])
        
        return [interpolated_out, interpolated_low_lims, interpolated_high_lims]

    #Decomp series takes low n_factors to high n_factors and stores lists of factors in a list within the DataTensor
    def decomposition_series(self, n_factors_low, n_factors_high, new_mz_len = None, gauss_params = None):
        self.decomps = []
        self.decomp_times = []
        for i in range(n_factors_low, n_factors_high):
            t1 = time.time()
            self.decomps.append(self.factorize(i, new_mz_len, gauss_params))
            t2 = time.time()
            self.decomp_times.append(t2-t1)
            print(str(i-n_factors_low+1)+" of "+str(n_factors_high-n_factors_low)+" T+"+str(t2-t1))
        
    #Takes single specified data type from a DataTensor object and desired number of factors, returns list of factorization components. Input lows and highs if 
    #using interpolated data, use n_concatenated to pass the number of data tensors concatenated together to the component object
    #Gauss_params must be tuple of len=2
    def factorize(self, n_factors, new_mz_len = None, gauss_params = None):
        
        factors=[]
        if self.n_concatenated != 1: 
            nnp = non_negative_parafac(self.concatenated_grid, n_factors, init = 'random')#, n_iter_max=50)
            for i in range(n_factors):
                factors.append(
                    Factor(
                        tensor_idx = self.tensor_idx, charge_state = self.charge_state, rts = nnp[1][0].T[i], dts = nnp[1][1].T[i], mz_data = nnp[1][2].T[i], factor_idx = i, 
                        n_factors = n_factors, lows = self.lows, highs = self.highs, abs_mz_low = self.mz_bin_low, n_concatenated = self.n_concatenated))
            return factors
            
        if new_mz_len != None:
            if gauss_params != None:
                grid, lows, highs = interpolate(self.full_grid_out, new_mz_len, gauss_params[0], gauss_params[1])
                nnp = non_negative_parafac(grid, n_factors, init = 'random')#, n_iter_max = 50)
            else: 
                grid, lows, highs = interpolate(self.full_grid_out, new_mz_len)
                nnp = non_negative_parafac(grid, n_factors, init = 'random')#, n_iter_max = 50)
        else: 
            lows, highs = self.lows, self.highs
            if gauss_params != None:
                grid = self.gauss(self.full_grid_out, gauss_params[0], gauss_params[1])
            else:
                grid = self.full_grid_out
                
        nnp = non_negative_parafac(grid, n_factors, init = 'random')#, n_iter_max = 50)
        for i in range(n_factors):
            factors.append(
                Factor(
                    tensor_idx = self.tensor_idx, charge_state = self.charge_state, rts = nnp[1][0].T[i], dts = nnp[1][1].T[i], mz_data = nnp[1][2].T[i], factor_idx = i, 
                    n_factors = n_factors, lows = lows, highs = highs, abs_mz_low = self.mz_bin_low, n_concatenated = self.n_concatenated))
        return factors
    
    
        
#Holds data from a single component of a non_negative_parafac factorization
#constructed as: (nnp[0].T[i], nnp[1].T[i], nnp[2].T[i], i, n, self.lows, self.highs, self.n_concatenated)
class Factor:
    
    def __init__(
            self, tensor_idx = None, charge_state = None, rts = None, dts = None, mz_data = None, factor_idx = None, 
            n_factors = None, lows = None, highs = None, abs_mz_low = None, n_concatenated = 1):

        self.tensor_idx = tensor_idx 
        self.charge_state = charge_state   
        self.rts = rts
        self.dts = dts
        self.mz_data = mz_data
        self.auc = sum(mz_data)
        self.factor_idx = factor_idx
        self.n_factors = n_factors
        self.lows = lows
        self.highs = highs
        self.n_concatenated = n_concatenated
        self.abs_mz_low = abs_mz_low
    
        #set up self.integrated_data
        self.integrated_mz_data = []
        self.grate = np.resize(False, (len(self.mz_data)))
        for i, j in zip(self.lows, self.highs):
            self.integrated_mz_data.append(sum(self.mz_data[i:j]))
            self.grate[i:j] = True       
        self.grate_sum = sum(self.mz_data[self.grate])
        
        self.max_rtdt = max(self.rts) * max(self.dts)
        self.outer_rtdt = sum(sum(np.outer(self.rts, self.dts)))
            
        #Peak error -> make this a method which can be optionally height filtered?
        #Implemented with normalization over values within integration boxes
        self.peak_error = 0
        
#This can be a shared function
        self.integration_box_centers = []
        for i, j in zip(self.lows, self.highs):
            self.integration_box_centers.append(i+((j-i)/2))
           
        self.box_intensities = self.mz_data[self.grate]
        self.max_peak_height = max(self.box_intensities)
        self.mz_peaks = sp.signal.find_peaks(self.mz_data, height=0.01)[0]
        self.peak_error, self.peaks_chosen=peak_error(self.mz_data, self.mz_peaks, self.integration_box_centers, self.max_peak_height)
                
        self.box_dist_avg=0
        for i in range(1,len(self.integration_box_centers)):
            self.box_dist_avg += self.integration_box_centers[i]-self.integration_box_centers[i-1]
        self.box_dist_avg = self.box_dist_avg/(len(self.integration_box_centers)-1)
                
        #Calls setter fxn
        self.find_isotope_clusters(5, height = 0.5)
            
        #Tuple of all relevant values and scores
        #Factor number, area under curve, grate sum, grate area ratio, peak error (box-norm), etc. as annotated or descriptively named below
        self.info_tuple = (self.tensor_idx,
                           self.factor_idx, 
                           self.auc,
                           self.grate_sum,
                           self.grate_sum/self.auc,
                           self.peak_error,
                           self.auc*self.max_rtdt,
                           self.auc*self.outer_rtdt
                          )

#This can call highs and lows from self instead of passing
    def find_isotope_clusters(self, peak_width, **kwargs):
        self.isotope_clusters=[]
        peaks = sp.signal.find_peaks(self.integrated_mz_data, **kwargs)[0]
        if len(peaks) == 0:
            return
        else:
            count = 0
            for i in range(len(peaks)):
                integrated_indices=find_window(self.integrated_mz_data, peaks[i], peak_width)
                if integrated_indices != None:
                    self.isotope_clusters.append(
                        IsotopeCluster(
                            self.charge_state, copy.deepcopy(self.mz_data), self.tensor_idx, self.n_factors, self.factor_idx, count, 
                            self.lows[integrated_indices[0]]-math.ceil(self.box_dist_avg/2), self.highs[integrated_indices[1]]+math.ceil(self.box_dist_avg/2), 
                            lows = self.lows, highs = self.highs, grate = self.grate, rts = self.rts, dts = self.dts, max_rtdt = self.max_rtdt, 
                            outer_rtdt = self.outer_rtdt, box_dist_avg = self.box_dist_avg, abs_mz_low = self.abs_mz_low))
                    count+=1
            return
    
#Constructed from peaks within integrated m/z space
class IsotopeCluster:
    
    def __init__(self, charge_state, factor_mz_data, tensor_idx, n_factors, factor_idx, cluster_idx, low_idx, high_idx, lows = None, 
                 highs=None, grate = None, rts = None, dts = None, max_rtdt = None, outer_rtdt = None, box_dist_avg = None, abs_mz_low = None):
        
        self.charge_state = charge_state
        self.factor_mz_data = factor_mz_data
        self.tensor_idx = tensor_idx
        self.n_factors = n_factors
        self.factor_idx = factor_idx
        self.cluster_idx = cluster_idx
        self.low_idx = low_idx
        self.high_idx = high_idx
        self.lows = lows
        self.highs = highs
        self.grate = grate
        self.rts = rts
        self.dts = dts
        self.max_rtdt = max_rtdt
        self.outer_rtdt = outer_rtdt
        self.box_dist_avg = box_dist_avg
        self.abs_mz_low = abs_mz_low
        
#This can be a function
        self.integration_box_centers = []
        for i, j in zip(self.lows, self.highs):
            self.integration_box_centers.append(i+((j-i)/2))
        
        self.cluster_mz_data = self.factor_mz_data
        self.cluster_mz_data[0:self.low_idx] = 0
        self.cluster_mz_data[self.high_idx:] = 0
        
        self.auc = sum(self.cluster_mz_data)
        
        #Values of isotope cluster that fall within the grate of expected peak bounds
        self.box_intensities = self.cluster_mz_data[self.grate]
        self.grate_sum = sum(self.box_intensities)
        
        self.mz_peaks = self.find_peaks(self.factor_mz_data, distance = self.box_dist_avg/10)[0]
        self.max_peak_height = max(self.box_intensities)
        self.peak_error, self.peaks_chosen = peak_error(self.cluster_mz_data, 
                                                        self.mz_peaks, 
                                                        self.integration_box_centers[np.searchsorted(self.lows, self.low_idx): np.searchsorted(self.highs, self.high_idx)], 
                                                        self.max_peak_height)
        
        self.baseline = peakutils.baseline(self.cluster_mz_data[self.low_idx:self.high_idx])
        self.baseline_subtracted_mz = self.cluster_mz_data
        self.baseline_subtracted_mz[self.low_idx: self.high_idx] = self.cluster_mz_data[self.low_idx: self.high_idx]-self.baseline
        
        self.baseline_auc = sum(self.baseline_subtracted_mz)
        
        self.baseline_box_intensities = self.baseline_subtracted_mz[self.grate]
        self.baseline_integrated_mz = []
        for lo, hi in zip(self.lows, self.highs):
            self.baseline_integrated_mz.append(sum(self.baseline_subtracted_mz[lo:hi]))
        self.abs_com = (np.flatnonzero(self.baseline_integrated_mz).mean()*self.box_dist_avg)+(self.lows[0]*0.002)+self.abs_mz_low
        self.baseline_grate_sum = sum(self.baseline_box_intensities)
        self.baseline_max_peak_height = max(self.baseline_box_intensities)
        self.baseline_peak_error, self.baseline_peaks_chosen = peak_error(self.baseline_subtracted_mz,
                                                                          self.mz_peaks,
                                                                          self.integration_box_centers[np.searchsorted(self.lows, self.low_idx): np.searchsorted(self.highs, self.high_idx)],
                                                                          self.baseline_max_peak_height)
                                                                         
        self.norm_fits = norm_fit(self.cluster_mz_data)
        
        self.info_tuple = (self.tensor_idx,
                           self.n_factors,
                           self.factor_idx, 
                           self.cluster_idx,
                           self.low_idx,
                           self.high_idx,
                           self.auc,
                           self.grate_sum,
                           self.grate_sum/self.auc,
                           self.baseline_auc,
                           self.baseline_grate_sum,
                           self.baseline_grate_sum/self.baseline_auc,
                           self.peak_error,
                           self.baseline_peak_error,
                           self.peak_error*(1-((1/(self.grate_sum/self.auc))**2)),
                           self.rts,
                           self.dts,
                           self.grate_sum*self.max_rtdt,
                           self.grate_sum*self.outer_rtdt,
                           self.norm_fits,
                           self.abs_com,
                           self.baseline_integrated_mz
                          )

#This is deprecated, just use sp.signal.find_peaks
        #Calculates peaks over whole source array, setter method allows for recalculation of peak_error after applying filtering to find_peaks
    def find_peaks(self, grid, **kwargs):
        self.mz_peaks = sp.signal.find_peaks(self.factor_mz_data, **kwargs)
        return self.mz_peaks
        
def norm_fxn(x, mean, mag, scal):
            return mag*(sp.stats.norm.pdf(x,loc=mean, scale=scal))
        
def norm_fit(array):
    p0=[np.argmax(array), max(array), len(array)/5]
    try:
        popt, pcov = curve_fit(norm_fxn, range(len(array)), array, p0)
        perr=np.sqrt(np.diag(pcov))
        norm_perr=[perr[0]/len(array), perr[1]/max(array), perr[2]/len(array)]
        return [popt, pcov, perr, norm_perr, p0]
    except:
        return None

def find_window(array, peak_idx, width):
        rflag = True
        lflag = True
        if peak_idx == 0:
            win_low = 0
            lflag = False
        if peak_idx == len(array)-1:
            win_high = len(array)-1
            rflag = False
            
        idx = peak_idx+1 
        if idx < len(array)-1: #Check if idx is last idx
            if array[idx] < array[peak_idx]/5: #Peak is likely not an IC if peak > 5 x neighbors
                win_high = idx
                rflag = False

        while rflag:
            #make sure looking ahead won't throw error
            if idx+1 < len(array):
                #if idx+1 goes down, and its height is greater than 20% of the max peak
                if array[idx+1] < array[idx] and array[idx+1] > array[peak_idx]/5:
                    idx += 1
                #if above check fails, test conditions separately
                else: 
                    if array[idx+1] < array[peak_idx]/5:
                        win_high = idx
                        rflag = False
                    else:
                        #first check if upward point is more than 5x the height of the base peak
                        if array[idx+1] < array[peak_idx]*5:
                            #look one point past upward point, if its below the last point and the next point continues down, continue
                            if idx+2 < len(array):
                                if array[idx+2] < array[idx+1]:
                                    if idx+3 < len(array):
                                        if array[idx+3] < array[idx+2]:
                                            idx += 3
                                        else: #point 3 ahead goes up, do not keep sawtooth tail
                                            win_high = idx
                                            rflag = False   
                                    else: #point 2 past idx is end of array, set as high limit
                                        win_high = idx+2
                                        rflag = False
                                else: #points continue increasing two ahead of idx, stop at idx
                                    win_high = idx
                                    rflag = False
                            else: #upward point is end of array, do not keep
                                win_high = idx
                                rflag = False
                        else: #upward point is major spike / base peak is minor, end search
                            win_high = idx
                            rflag = False
            else: #idx is downward and end of array
                    win_high = idx
                    rflag = False        

        idx=peak_idx-1
        if idx >= 0:
            if array[idx] < array[peak_idx]/5:
                win_low=idx
                lflag=False

        while lflag:
            if idx-1 >= 0: #make sure looking ahead won't throw error
                if array[idx-1] < array[idx] and array[idx-1] > array[peak_idx]/5: #if idx-1 goes down, and its height is greater than 20% of the max peak
                    idx -= 1
                #if above check fails, test conditions separately
                else:
                    if array[idx-1] < array[peak_idx]/5:
                        win_low = idx
                        lflag = False
                    else:
                        #first check if upward point is more than 5x the height of the base peak
                        if array[idx-1] < array[peak_idx]*5:
                            #look one point past upward point, if its below the last point and the next point continues down, continue
                            if idx-2 >= 0:
                                if array[idx-2] < array[idx]:
                                    if idx-3 >= 0:
                                        if array[idx-3] < array[idx-2]:
                                            idx -= 3
                                        else:  #point 3 ahead goes up, do not keep sawtooth tail
                                            win_low = idx
                                            lflag = False
                                    else: #point 2 past idx is end of array, set as high limit
                                        win_low = idx-2
                                        lflag = False
                                else: #points continue increasing two ahead of idx, stop at idx
                                    win_low = idx
                                    lflag = False
                            else: #upward point is start of array, do not keep
                                win_low = idx
                                lflag = False
                        else: #upward point is major spike / base peak is minor, end search
                            win_low = idx
                            lflag = False
            else: #idx is start of array
                win_low = idx
                lflag = False    
                    
        if win_high-win_low < width:
            return None
        else:
            return [win_low, win_high]

#Allows for recalculation of peak_error after filtering peak selection
def peak_error(source, mz_peaks, integration_box_centers, max_peak_height):
    peak_error = 0
    peaks_chosen = []
    peaks_total_height = 0
    match_idx = np.searchsorted(mz_peaks, integration_box_centers)
    if len(mz_peaks) > 0:
        for i in range(len(match_idx)):
            #handle peaks list of length 1
            if len(mz_peaks) == 1:
                peak_error += abs(integration_box_centers[i]-mz_peaks[0])*(source[mz_peaks[0]])
                peaks_chosen.append(mz_peaks[0])
                peaks_total_height += source[mz_peaks[0]]
            else:                  
                #check if place to be inserted is leftmost of peaks
                if match_idx[i] == 0:
                    peak_error += abs(integration_box_centers[i]-mz_peaks[match_idx[i]])*(source[mz_peaks[match_idx[i]]])
                    peaks_chosen.append(mz_peaks[match_idx[i]])
                    peaks_total_height += source[mz_peaks[match_idx[i]]]
                else:
                    #check if insertion position is rightmost of peaks
                    if match_idx[i] == len(mz_peaks):
                        peak_error += abs(integration_box_centers[i]-mz_peaks[-1])*(source[mz_peaks[-1]])
                        peaks_chosen.append(mz_peaks[-1])
                        peaks_total_height += source[mz_peaks[-1]]
                    else:
                        #handle case where distances between peaks are the same, pick biggest peak
                        if abs(integration_box_centers[i]-mz_peaks[match_idx[i]]) == abs(integration_box_centers[i]-mz_peaks[match_idx[i]-1]):
                            peak_error += max([abs(integration_box_centers[i]-mz_peaks[match_idx[i]])*(source[mz_peaks[match_idx[i]]]),
                                        abs(integration_box_centers[i]-mz_peaks[match_idx[i]-1])*(source[mz_peaks[match_idx[i]-1]])])
                            if (abs(integration_box_centers[i]-mz_peaks[match_idx[i]])*(source[mz_peaks[match_idx[i]]]) >
                                abs(integration_box_centers[i]-mz_peaks[match_idx[i]-1])*(source[mz_peaks[match_idx[i]-1]])):
                                peaks_chosen.append(mz_peaks[match_idx[i]])
                                peaks_total_height += source[mz_peaks[match_idx[i]]]
                            else:
                                peaks_chosen.append(mz_peaks[match_idx[i]-1])
                                peaks_total_height += source[mz_peaks[match_idx[i]-1]]
                        else:
                            #only need to check left hand side differences because of left-hand default of searchsorted algorithm
                            #now check which peak is closer, left or right. This poses problems as there may be very close peaks which are not
                            #actually significant in height but which pass filtering. This penalizes less noisy components, and advantages 
                            #components with noise just above the height filter level. Consider noise removal step, or use high number of factors in parafac
                            if abs(integration_box_centers[i]-mz_peaks[match_idx[i]]) < abs(integration_box_centers[i]-mz_peaks[match_idx[i]-1]):
                                peak_error += abs(integration_box_centers[i]-mz_peaks[match_idx[i]])*(source[mz_peaks[match_idx[i]]])
                                peaks_chosen.append(mz_peaks[match_idx[i]])
                                peaks_total_height += source[mz_peaks[match_idx[i]]]
                            else:
                                peak_error += abs(integration_box_centers[i]-mz_peaks[match_idx[i]-1])*(source[mz_peaks[match_idx[i]-1]])
                                peaks_chosen.append(mz_peaks[match_idx[i]-1])
                                peaks_total_height += source[mz_peaks[match_idx[i]-1]]
        
        box_dist_total = 0
        for i in range(1,len(integration_box_centers)):
            box_dist_total += integration_box_centers[i]-integration_box_centers[i-1]
        
        
        peak_error = (peak_error / peaks_total_height / (box_dist_total/(len(integration_box_centers)-1)))
        return peak_error, peaks_chosen