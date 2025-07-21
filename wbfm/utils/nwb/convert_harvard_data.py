import h5py
import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.data_utils import DataChunkIterator
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from pynwb.ophys import Fluorescence, RoiResponseSeries
from pynwb.behavior import Position, SpatialSeries
from ndx_multichannel_volume import MultiChannelVolumeSeries
from wbfm.utils.nwb.utils_nwb_export import build_optical_channel_objects, _zimmer_microscope_device
from pynwb.image import ImageSeries
from pynwb import TimeSeries
from datetime import datetime
from dateutil.tz import tzlocal
from wbfm.utils.nwb.utils_nwb_export import CustomDataChunkIterator
import dask.array as da


def iter_frames(h5_file, n_timepoints, frame_shape):
    for i in range(n_timepoints):
        data = h5_file[str(i)]["frame"]
        yield da.from_array(data, chunks=frame_shape).transpose((1, 2, 3, 0))  # Lazy loading

def dask_stack_volumes(volume_iter):
    """Stack a generator of volumes into a dask array along time."""
    return da.stack(volume_iter, axis=0)  


# === USER PARAMETERS ===
input_path = "185.h5"
output_path = "185.nwb"
experiment_name = input_path.split("/")[-1].split(".")[0]
frame_rate = 10.0  

with h5py.File(input_path, "r") as f:

    # === Create NWBFile ===
    nwbfile = NWBFile(
        session_description="Freely moving whole-brain imaging",
        identifier=experiment_name,
        session_start_time=datetime.now(tz=tzlocal()),
        experimenter="Harvard Guy "+str(experiment_name),
        lab="Some Harvard Lab",
        institution="Harvard",
    )

    # === Add Subject ===
    nwbfile.subject = Subject(
        subject_id="subject "+experiment_name,
        species="Caenorhabditis elegans",
        description="Tiny worm",
    )



    # === Add Fluorescence Data ===
    ci_int = f["ci_int"][:]  # shape (97, 1331, 12)
    ci_mean = ci_int.mean(axis=2)  # shape (97, 1331) → average over 12 pixels

    # Transpose to (time, neurons)
    ci_mean = ci_mean.T  # shape (1331, 97)

    """flu = Fluorescence()
    flu.add_roi_response_series(RoiResponseSeries(
        name="mean_fluorescence",
        data=ci_mean,
        rois=, 
        unit="a.u.",
        rate=frame_rate,
        description="Mean calcium intensity (ΔF/F) per neuron, averaged over 12 ROI pixels",
    ))
    
    nwbfile.processing["ophys"] = flu"""

    flu_ts = TimeSeries(
    name="mean_fluorescence",
    data=ci_mean,
    unit="a.u.",
    rate=frame_rate,
    description="Mean calcium intensity per neuron, averaged over ROI pixels"
    )
    nwbfile.add_acquisition(flu_ts)

    


    """# === Add Neuron Presence Mask ===
    neuron_presence = f["neuron_presence"][:]  # (1331, 98)

    nwbfile.add_acquisition(TimeSeries(
        name="neuron_presence",
        data=neuron_presence,
        unit="bool",
        rate=frame_rate,
        description="Is the neuron currently present in the image",
    ))"""


    # === Add 3D Tracking Data ===
    calcium_imaging_module = nwbfile.create_processing_module(
        name='CalciumActivity',
        description='Calcium time series metadata, segmentation, and fluorescence data'
    )
    points = f["points"][:]  # shape (1331, 98, 3)
    position_module = Position(name="NeuronCentroids")

    for neuron_idx in range(points.shape[1]):
        neuron_trace = points[:, neuron_idx, :]  # (1331, 3)
        print(neuron_trace.shape)

        position_module.add_spatial_series(SpatialSeries(
            name=f"neuron_{(neuron_idx+1):03d}",
            data=neuron_trace,
            unit="micrometers",
            reference_frame="Lab frame",
            description=f"3D position of neuron {neuron_idx}",
            timestamps=np.arange(neuron_trace.shape[0])
        ))

    calcium_imaging_module.add(position_module)

    frame_shape = (320, 192, 20, 2) #np.transpose(f["0/frame"].shape, (1,2,3,0))  # (2, 320, 192, 20)
    chunk_shape = (1,) + frame_shape

    nn_keys = []
    for key in f.keys():
        if key.isdigit():
            nn_keys.append(int(key))
    num_frames = np.array(nn_keys).max() + 1
    print(num_frames)
    
    series_shape = (num_frames, 320, 192, 20, 2)
    dtype = f["0/frame"].dtype

    
    # Build metadata objects
    grid_spacing = (0.3, 0.3, 0.3)  # Flavell data is isotropic
    device = _zimmer_microscope_device(nwbfile)
    CalcImagingVolume, _ = build_optical_channel_objects(device, grid_spacing, ['red', 'green'])
    # Add directly to the file to prevent hdmf.build.errors.OrphanContainerBuildError
    nwbfile.add_imaging_plane(CalcImagingVolume)


    imvol_dask = dask_stack_volumes(iter_frames(f,num_frames, frame_shape))
    chunk_video = (1,) + imvol_dask.shape[1:-1] + (1,)
    video_data = H5DataIO(
        data=CustomDataChunkIterator(array=imvol_dask, chunk_shape=chunk_video, display_progress=True),
        compression="gzip"
    )


    nwbfile.add_acquisition(MultiChannelVolumeSeries(
        name="CalciumImageSeries",
        description="Series of calcium imaging data",
        comments="Calcium imaging data from Harvard lab",
        data=video_data,  # data here should be series of indexed masks
        # Elements below can be kept the same as the CalciumImageSeries defined above
        device=device,
        unit="Voxel gray counts",
        scan_line_rate=2995.,
        dimension=series_shape,  
        resolution=1.,
        # smallest meaningful difference (in specified unit) between values in data: i.e. level of precision
        rate=frame_rate,  # sampling rate in hz
        imaging_volume=CalcImagingVolume,
    ))


    # === Write NWB file ===
    with NWBHDF5IO(output_path, "w") as io:
        io.write(nwbfile)

print(f"NWB file written to {output_path}")
