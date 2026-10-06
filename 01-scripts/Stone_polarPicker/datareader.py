import os
import numpy as np
import h5py
import pandas as pd
import obspy
from obspy import UTCDateTime
from scipy import signal as s
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical

class DataReader:
    def __init__(self, format="numpy", sampling_rate=100, **kwargs):
        self.buffer = {}
        self.format = format
        self.sampling_rate = sampling_rate

        if format == "numpy":
            self.data_dir = kwargs["data_dir"]
            try:
                csv = pd.read_csv(kwargs["data_list"], header=0,sep="[,|\s+]", engine="python")
            except:
                csv = pd.read_csv(kwargs["data_list"], header=0, sep="\t")
            self.data_list = csv["fname"]
            self.num_data = len(self.data_list)
        #elif format == "hdf5":
        #    self.h5 = h5py.File(kwargs["hdf5_file"], "r", libver="latest")
        #    self.h5_data
        elif format == "mseed":
            self.data_dir = kwargs["data_dir"]
            try:
                csv = pd.read_csv(kwargs["data_list"], header=0,sep="[,|\s+]", engine="python")
            except:
                csv = pd.read_csv(kwargs["data_list"], header=0)
            self.evids = csv["evid"]
            self.stas = csv["sta"]
            self.tphases = csv["UTCDateTime"]
        else:
            raise (f"{format} not supported")
    
    def read_numpy(self,fname):
    
        if fname not in self.buffer:
            npz = np.load(fname)
            meta = {}
            meta["data"] = npz["data"][:,2] # Z-comp
            meta["p_idx"] = npz["p_idx"]
        else:
            meta = self.buffer[fname]

        return meta
    
    def read_mseed(self,fname,tphase):

        if fname not in self.buffer:
            meta = {}
            msd = obspy.read(fname)
            if len(msd)<3:
                meta["inrange"] = False
                return meta
            meta["data"] = msd[2].data
            p_cnt = int((UTCDateTime(tphase) - UTCDateTime(msd[2].stats.starttime))/msd[2].stats.delta)
            if p_cnt > msd[2].stats.npts:
                #raise ValueError(f"Calculated phase index for {fname} is greater than record length")
                meta["inrange"] = False
            else:
                meta["inrange"] = True
            meta["p_idx"] = p_cnt
        else:
            meta = self.buffer[fname]

        return meta
    
class DataReader_train(DataReader): 
    def __init__(self, format="numpy",**kwargs):
        super().__init__(format=format,**kwargs)

    def get_numpy_data(self):
    #def __getitem__(self):

        base_name = self.data_list

        Xarr = []
        yarr = []

        pbar = tqdm(total=len(base_name))

        for rec in base_name:
            pbar.update()
            if self.format == "numpy":
                meta = self.read_numpy(os.path.join(self.data_dir,rec))
            data = meta["data"]
            p_idx = meta["p_idx"]
            Xarr.append(data[p_idx-32:p_idx+32])
            #yarr.append() # need to read in polarity values somewhere 

        #return (np.array(Xarr), to_categorical((np.array(yarr) == 'positive').astype(int)))
        return (np.array(Xarr))
    
    def get_mseed_data(self):

        evids = self.evids
        stas = self.stas
        tphases = self.tphases

        rec_list = np.zeros(len(evids),dtype='<U50')
        Xarr = np.zeros((len(evids),64))
        SNRarr = np.zeros(len(evids))
        rec_bool = np.zeros(len(evids),dtype=bool)

        pbar = tqdm(total=len(evids))

        for i in range(len(evids)):
            pbar.update()
            rec = evids[i] + "_" + stas[i] + ".mseed"
            if os.path.isfile(os.path.join(self.data_dir,rec)):
                tphase = tphases[i]
                if self.format == "mseed":
                    meta = self.read_mseed(os.path.join(self.data_dir,rec),tphase)
                if meta["inrange"] == False:
                   continue
                data = meta["data"]
                p_idx = meta["p_idx"]
                if(len(data[p_idx-32:p_idx+32])<64):
                   continue
                signal = data[p_idx:p_idx+25].copy()
                noise = data[p_idx-50:p_idx].copy()
                if(np.max(signal)-np.min(signal)==0.):
                    continue
                if((len(signal)<25)|(len(noise)<50)):
                    SNR = 0.1
                else:
                    signal -= np.arange(len(signal))/len(signal)*(signal[-1]-signal[0])/0.25+signal[0]
                    noise -= np.arange(len(noise))/len(noise)*(noise[-1]-noise[0])/0.5+noise[0]
                    signal_std = np.std(signal)
                    noise_std = np.std(noise)
                    SNR = signal_std/noise_std
                SNRarr[i] = SNR
                rec_list[i] = rec
                Xarr[i] = data[p_idx-32:p_idx+32]
                rec_bool = True
                
        Xarr = Xarr[rec_bool,:]
        rec_list = rec_list[rec_bool]
        SNRarr = SNRarr[rec_bool]
        
        return(Xarr,list(rec_list),SNRarr)
    
    def get_mseed_data_filtered(self,fmin,fmax):

        dt = self.sampling_rate
        evids = self.evids
        stas = self.stas
        tphases = self.tphases

        rec_list = []
        Xarr = []
        SNRarr = []

        sost = s.butter(2,[fmin,fmax],'bandpass',fs=dt,output='sos')

        pbar = tqdm(total=len(evids))

        for i in range(len(evids)):
            pbar.update()
            rec = evids[i] + "_" + stas[i] + ".mseed"
            if os.path.isfile(os.path.join(self.data_dir,rec)):
                tphase = tphases[i]
                if self.format == "mseed":
                    meta = self.read_mseed(os.path.join(self.data_dir,rec),tphase)
                if meta["inrange"] == False:
                   continue
                data = meta["data"]
                p_idx = meta["p_idx"]
                if(len(data[p_idx-32:p_idx+32])<64):
                   continue
                data = s.sosfiltfilt(sost,data)
                signal = data[p_idx:p_idx+100].copy()
                noise = data[p_idx-200:p_idx].copy()
                if((len(signal)<100)|(len(noise)<200)):
                    SNR = 0.5
                else:
                    signal -= np.arange(len(signal))/len(signal)*(signal[-1]-signal[0])/1.+signal[0]
                    noise -= np.arange(len(noise))/len(noise)*(noise[-1]-noise[0])/2.+noise[0]
                    signal_std = np.std(signal)
                    noise_std = np.std(noise)
                    SNR = signal_std/noise_std
                SNRarr.append(SNR)
                rec_list.append(rec)
                Xarr.append(data[p_idx-32:p_idx+32])


        return(np.array(Xarr),rec_list,np.array(SNRarr))

    def calc_mseed_SNR(self):
        evids = self.evids
        stas = self.stas
        tphases = self.tphases

        SNR_arr = []

        pbar = tqdm(total=len(evids))

        for i in range(len(evids)):
            pbar.update()
            rec = evids[i] + "_" + stas[i] + ".mseed"
            # employs the same discrimination metric as get_mseed_data()
            if os.path.isfile(os.path.join(self.data_dir,rec)):
                tphase = tphases[i]
                if self.format == "mseed":
                    meta = self.read_mseed(os.path.join(self.data_dir,rec),tphase)
                if meta["inrange"] == False:
                   continue
                data = meta["data"]
                p_idx = meta["p_idx"]
                if(len(data[p_idx-32:p_idx+32])<64):
                   continue
                signal = data[p_idx:p_idx+100]
                noise = data[p_idx-200:p_idx]
                if((len(signal)<100)|(len(noise)<200)):
                    SNR = 0.5
                else:
                    signal -= np.arange(len(signal))/len(signal)*(signal[-1]-signal[0])/1.+signal[0]
                    noise -= np.arange(len(noise))/len(noise)*(noise[-1]-noise[0])/2.+noise[0]
                    signal_std = np.std(signal)
                    noise_std = np.std(noise)
                    SNR = signal_std/noise_std
                SNR_arr.append(SNR)

        return(np.array(SNR_arr))

