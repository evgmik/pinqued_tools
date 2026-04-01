''''
Class for FLIR camera data acquisition

Author: Mykhailo Vorobiov
'''
#%%
import numpy as np
import os
from time import perf_counter
from datetime import datetime
from typing import Literal, Dict, Any, Tuple, Optional
from numpy.typing import NDArray

import PySpin

# External dependencies 
# (assuming these are available in the environment)
from simple_pyspin import Camera


class FLIRCamera():
    '''
    Class for FLIR camera data acquisition and control.
    '''
    TRIGGER_ON_OFF = ['Off', 'On']


    def __init__(self, 
                 cam_index: int = 0,
                 exposure_value: float = 40000.0,
                 gain_value: float = 20.0
                 ):
        
        self._cam_index = cam_index

        # 1. Get singleton reference to system object
        self._system = PySpin.System.GetInstance()

        # 2. Get camera list
        self._cam_list = self._system.GetCameras()
        num_cameras = self._cam_list.GetSize()

        if num_cameras == 0:
            self._cam_list.Clear()
            self._system.ReleaseInstance()
            raise AssertionError('No cameras detected!')
        
        # 3. Get camera and initialize
        self._cam = self._cam_list.GetByIndex(cam_index)
        self._cam.Init()

        try:
            self._cam.Init()

            # Assuming cam is an initialized PySpin Camera object
            node_map = self._cam.GetNodeMap()

            # 1. Turn off Auto Exposure
            exposure_auto = PySpin.CEnumerationPtr(node_map.GetNode('ExposureAuto'))
            exposure_auto.SetIntValue(exposure_auto.GetEntryByName('Off').GetValue())

            # Set manual exposure time (in microseconds)
            exposure_time = PySpin.CFloatPtr(node_map.GetNode('ExposureTime'))
            exposure_time.SetValue(exposure_value) 

            # 2. Turn off Auto Gain
            gain_auto = PySpin.CEnumerationPtr(node_map.GetNode('GainAuto'))
            gain_auto.SetIntValue(gain_auto.GetEntryByName('Off').GetValue())

            # Set manual gain (in dB)
            gain = PySpin.CFloatPtr(node_map.GetNode('Gain'))
            gain.SetValue(gain_value) # Example: 0 dB

            # 3. Disable Gamma
            gamma_enable = PySpin.CBooleanPtr(node_map.GetNode('GammaEnable'))
            gamma_enable.SetValue(False) # Set to False to disable, True to enable

            # 1. Enable Chunk Mode
            self._cam.ChunkModeActive.SetValue(True)

            # 2. Enable Timestamp Chunk
            self._cam.ChunkSelector.SetValue(PySpin.ChunkSelector_Timestamp)
            self._cam.ChunkEnable.SetValue(True)

            print('Camera initialized')
        except PySpin.SpinnakerException as ex:
            print(f'Error: {ex}')
    
    def switch(self, cam_index: int):
        if self._cam is not None:
            self._cam.DeInit()
            del self._cam
        if self._cam_index != cam_index:
            self._cam_index = cam_index
            self._cam = self._cam_list.GetByIndex(cam_index)

    def deinit(self):
        self._cam.DeInit()
        del self._cam
        self._cam_list.Clear()
        self._system.ReleaseInstance()
        print('Camera deinitialized')
        
    def trigger(self, on: bool = True):
        if on:
            self._cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
            self._cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0) # Or Line0-2 depending on hardware
            self._cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
            self._cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        else:
            if self._cam.IsStreaming():
                self._cam.EndAcquisition()
            self._cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
            self._cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_MultiFrame)
    
    def acquire_sequence(self, num_frames: int):
        # 1. Enable chunk data
        nodemap = self._cam.GetNodeMap()
        chunk_mode = PySpin.CBooleanPtr(nodemap.GetNode('ChunkModeActive'))
        if chunk_mode.GetAccessMode() == PySpin.RW:
            chunk_mode.SetValue(True)

        # Enable timestamp
        chunk_selector = PySpin.CEnumerationPtr(nodemap.GetNode('ChunkSelector'))
        chunk_entry = chunk_selector.GetEntryByName('Timestamp')
        chunk_selector.SetIntValue(chunk_entry.GetValue())
        chunk_enable = PySpin.CBooleanPtr(nodemap.GetNode('ChunkEnable'))
        chunk_enable.SetValue(True)
        print('Timestamp chunk enabled')

        # 2. Acquire images and get timestamp
        self._cam.BeginAcquisition()
        print(f'Camera acquisition started. Acquiring {num_frames} frames...')

        timestamps_ms = np.zeros(num_frames)
        for i in range(num_frames):
            image = self._cam.GetNextImage()

            # Get the chunk data container
            chunk_data = image.GetChunkData()
            # Get timestamp in nanoseconds
            timestamp = chunk_data.GetTimestamp()

            # 3. Calculate difference (ns to ms)
            timestamp_ms = timestamp / 1e6
            print(f"Time stamp: {timestamp_ms:.4f} ms")

            timestamps_ms[i] = timestamp_ms
            image.Release()

        self._cam.EndAcquisition()
        timestamps_ms = timestamps_ms - timestamps_ms[0]
        print(timestamps_ms)
        print('Acquisition ended')

    def disable_fps_limit(self):
        if hasattr(self._cam, 'AcquisitionFrameRateEnable'):
            self._cam.AcquisitionFrameRateEnable.SetValue(False)
        elif hasattr(self._cam, 'AcquisitionFrameRateEnabled'):
            self._cam.AcquisitionFrameRateEnabled.SetValue(False)

    def end_acqusition(self):
        self._cam.EndAcquisition()
        print('Acquisition ended')


#%%
# -------------- Usage example -------------
if __name__=='__main__':
    # Apply custom plotting style
    from matplotlib import pyplot as plt

    cam = FLIRCamera(cam_index=0,
                     exposure_value=10000)
    cam.disable_fps_limit()
    cam.trigger(on=True)
    cam.acquire_sequence(10)
    cam.deinit()
    

# %%
