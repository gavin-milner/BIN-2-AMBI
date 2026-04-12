# This program extracts features from the binaural audio (ITD, IPD, ILD and rotation features)
# and stores them to the dataset.

import sys
import numpy as np
import librosa
import torch
import pandas as pd
import re

from pathlib import Path
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R
from mel_filtering import MelFeatureMapper

# Get project root directory
ROOT = Path(__file__).resolve().parents[2]

# Add path to sys to access the binaspect module
binaspect_path = Path(ROOT / "modules") / "binaspect"
sys.path.append(str(binaspect_path))
import binaspect

class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.sr = config["analysis"]["sample_rate"]
        self.window_size = config["analysis"]["window_size"]
        self.overlap = config["analysis"]["overlap"]
        self.start = config["analysis"]["start_freq"]
        self.stop = config["analysis"]["stop_freq"]
        self.to_extract = config["analysis"]["extract"]
        self.file_type = config["generation"]["file_type"]

        self.n_mels = config["analysis"].get("mel", {}).get("n_mels", 64)
        self.rotation_flag = None
        self.hop_size = round(self.window_size * self.overlap) 
        self.mel_flag = self.config["analysis"]["mel"]["active"]

        # Store features here to be able to be acessed later
        self.feats = None
        self.metadata = []

        # Store features directly to Google Drive?
        if self.config["paths"].get("save_to_drive", False):
            drive_path = Path(self.config["paths"].get("google_drive_path", ""))
            SAVE_ROOT = drive_path
        else:
            SAVE_ROOT = ROOT
                
        self.binaural_dir = ROOT / "datasets" / self.config["dataset_name"] / self.config["paths"]["binaural_folder"]
        self.features_dir = SAVE_ROOT / "datasets" / self.config["dataset_name"] / self.config["paths"]["features_folder"]

        # Check rotation flags from config
        self.analyze_rot0 = self.config["analysis"]["rotation"].get("rot0", True)
        self.analyze_rot1 = self.config["analysis"]["rotation"].get("rot1", True)

        self.alpha = self.config["analysis"].get("alpha", None)

    def run(self, q = None, mode = 'test'):

        if mode == 'pipe' and q is not None:
        # Live processing from queue if 'pipe' mode is activated
            pbar = tqdm(desc="Processing incoming Binaural Files", unit="file")
            while True:
                print(q)
                item = q.get()
                if item is None:
                    break
                self.process_file(item) # process the file
                pbar.update(1)
            pbar.close()

        else:
        # Or process existing dataset if already generated:
            files = sorted((f for f in self.binaural_dir.iterdir() if f.suffix == self.file_type),
                           key=lambda f: int(re.search(r'\d+', f.stem).group()))
            
            for f in tqdm(files, desc="Extracting Binaural Features", unit="file"):
                self.process_file(f)
            
            # Save metadata csv once all files have been processed
            if self.config.get("metadata", {}).get("save_csv", False):
                metadata_df = pd.DataFrame(self.metadata)
                metadata_csv_path = self.features_dir / "metadata.csv"
                metadata_df.to_csv(metadata_csv_path, index=False)

    def process_file(self, file):
        # Convert file back to Path object if it arrives as a string
        if isinstance(file, str):
            file = Path(file)

        # Skip files based on rotation flags
        if "rot_0" in file.stem and not self.analyze_rot0:
            return
        if "rot_0" not in file.stem and not self.analyze_rot1:
            return

        # Add rotation info to metadata
        self.rotation_flag = "rot_0" if "rot_0" in file.stem else "rot_1"

        # Load the audio scene using librosa
        scene_path = self.binaural_dir / file
        audio, _ = librosa.load(scene_path, sr=self.sr, mono=False)

        # Extract features
        feats = self.extract_features(audio)
        self.feats = feats
        # Also extract rotation features if they are present
        if self.rotation_flag == "rot_1":
            rotation_feats = self.extract_rotation_features(audio)
            feats.update(rotation_feats)
        scene_name = file.stem     

        # Save as .pt file             
        #feature_file = self.features_dir / f"{scene_name}.pt"
        #torch.save(feats, feature_file)

        # Save as numpy files
        feature_file = self.features_dir / f"{scene_name}.npz"
        np.savez(feature_file, **feats)

        self.metadata.append({
            "scene": scene_name,
            "rotation": self.rotation_flag,
            "feature_file": feature_file,
            "cues": ",".join(feats.keys())
        })

    # Function to apply the mel mapping
    def map_mel(self, data):
        mel_filter = MelFeatureMapper(sr=self.sr, bin_size=data.shape[0], n_mels=self.n_mels)
        return mel_filter.map(data)

    # Compute the binaural features:
    def extract_features(self, audio):
        feats = {}

        # Extract ITD features
        if self.to_extract.get("ITD", False):
            ITD = binaspect.ITD_spect(audio, self.sr, 
                                      window_size = self.window_size,
                                      overlap = self.overlap,
                                      start_freq=self.start, 
                                      stop_freq=self.stop, 
                                      plots=False)
            # Map to mel feature and add to features
            # Initalise the mel_filter
            #print("ITD Shape: ", ITD.shape)
            if self.mel_flag:
                feats["ITD"] = self.map_mel(ITD)
            else:
                feats["ITD"] = ITD

        # Extract ILD features
        if self.to_extract.get("ILD", False):
            ILD = binaspect.ILD_spect(audio, self.sr,  
                                      window_size = self.window_size,
                                      overlap = self.overlap,
                                      start_freq=self.start, 
                                      stop_freq=self.stop, 
                                      plots=False)
            #print("ILD Shape: ", ILD.shape)
            if self.mel_flag:
                feats["ILD"] = self.map_mel(ILD)
            else:
                feats["ILD"] = ILD

        # Extract IPD features
        if self.to_extract.get("IPD", False):
            IPD_sine = binaspect.IPD_spect_custom(audio, self.sr, 
                                      start_freq=self.start,
                                      window_size = self.window_size,
                                      overlap = self.overlap, 
                                      stop_freq=self.stop, 
                                      mode="sin",
                                      wrapped=False, plots=False)
            #print("IPD Shape: ", IPD.shape)
            if self.mel_flag:
                feats["IPD_sine"] = self.map_mel(IPD_sine)
            else:
                feats["IPD_sine"] = IPD_sine

            IPD_cosine = binaspect.IPD_spect_custom(audio, self.sr, 
                                      start_freq=self.start,
                                      window_size = self.window_size,
                                      overlap = self.overlap, 
                                      stop_freq=self.stop, 
                                      mode="cos",
                                      wrapped=False, plots=False)
            #print("IPD Shape: ", IPD.shape)
            if self.mel_flag:
                feats["IPD_cosine"] = self.map_mel(IPD_cosine)
            else:
                feats["IPD_cosine"] = IPD_cosine  

        if self.to_extract.get("MEAN_MAG", False):
            MEAN_MAG = binaspect.mean_mag_spect(
                audio, self.sr,
                start_freq=self.start,
                window_size=self.window_size,
                overlap=self.overlap,
                stop_freq=self.stop,
                log_scale=True,
                alpha = self.alpha,
                plots=False
            )

            if self.mel_flag:
                feats["mean_mag"] = self.map_mel(MEAN_MAG)
            else:
                feats["mean_mag"] = MEAN_MAG  

        # Extract IC features
        if self.to_extract.get("IC", False):
            IC = binaspect.IC_spect(audio, self.sr, 
                                      start_freq=self.start,
                                      window_size = self.window_size,
                                      overlap = self.overlap, 
                                      stop_freq=self.stop, 
                                      alpha = self.alpha,
                                      plots=False)
            #print("IC Shape: ", IC.shape)
            if self.mel_flag:
                feats["IC"] = self.map_mel(IC)
            else:
                feats["IC"] = IC

        return feats

    # Extract the rotation features if they are present:
    def extract_rotation_features(self, audio: np.ndarray) -> dict:
        feats = {}
        if audio.shape[0] < 5:
            return feats

        # The head tracking information is in channel 3 - 5:
        acc_data = audio[2:5, :] 
        #n_samples = acc_data.shape[1] # Find the number of samples

        # Pad the window out as in librosa stft:
        pad = self.window_size // 2
        acc_data_padded = np.pad(acc_data, ((0,0), (pad, pad)), mode='constant', constant_values=0)
        n_samples_padded = acc_data_padded.shape[1]

        # Loop through audio and get the average rotation features per frame:
        frames = []
        for start in range(0, n_samples_padded - self.window_size + 1, self.hop_size):
            window = acc_data_padded[:, start:start+self.window_size] # window frame has 3 channels 
            # Get the average in the window across the columns (across time) for each channel:
            frame_avg = window.mean(axis=1)  
            # Add it to the list of frames               
            frames.append(frame_avg)

        # Stack frames to get 3 X T size frame:
        rot_frames_mean = np.stack(frames, axis=1) 

        # Convert back from normalised to radians
        #rot_frames_mean = rot_frames_mean * np.pi
        rot_frames_mean = rot_frames_mean

        # Convert to quaternions
        quaternions = []
        # Loop through the frames that were created
        for t in range(rot_frames_mean.shape[1]):
            # Create a rotation object in the order of the angles
            #r = R.from_euler('zyx', rot_frames_mean[:, t])
            r = R.from_euler('xyz', rot_frames_mean[:, t])
            q = r.as_quat()  # Convert to quaternions x, y, z, w
            q = np.array([q[3], q[0], q[1], q[2]])  # Reorder to w, x, y, z
            quaternions.append(q)
        
        # Convert list of arrays into matrix
        quaternions = np.stack(quaternions, axis=0).T  # 4 X T
        # Change to 3D tensor to align with other features
        #print(quaternions[:, :1])
        feats["rotation"] = quaternions
        print("Rotation Shape: ", quaternions.shape)

        return feats
