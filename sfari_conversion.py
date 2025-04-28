import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import dateutil
import uuid
from datetime import datetime
import pytz

import pynwb
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from ndx_franklab_novela import CameraDevice
from ndx_pose import PoseEstimationSeries, PoseEstimation, Skeleton


def read_data(animal_dir: Path):
    """
    Read in the data files
    """

    animal_dir = Path("/cumulus/sam/safri_data/frank/session_20240731094729")
    moseq_df_path = animal_dir.parent / "moseq_df.csv"

    # metadata
    mertadata_path = animal_dir / "metadata.json"
    metadata_df = pd.read_json(mertadata_path, lines=True)
    metadata_df["StartTime"] = dateutil.parser.parse(
        metadata_df.StartTime.values[0]
    ).timestamp()

    # Video data
    video_paths = [x for x in animal_dir.iterdir() if x.suffix in [".mp4", ".avi"]]
    video_timestamps = animal_dir / "depth_ts.txt"
    vid_timestamps_df = pd.read_csv(
        video_timestamps, header=None, names=["timestamp (s)"]
    )
    vid_timestamps_df["timestamp (s)"] = (
        vid_timestamps_df["timestamp (s)"].astype(float) / 1000
        + metadata_df["StartTime"].values[0]
    )

    # Moseq df (contains centroid, orientation, velocities as well)
    moseq_df = pd.read_csv(
        moseq_df_path,
    )
    contain_vals = ["arid1b", "20240731092547"]
    moseq_name = [
        x for x in moseq_df.name.unique() if all([y in x for y in contain_vals])
    ]
    moseq_df = moseq_df[moseq_df.name == moseq_name[0]]
    moseq_df["timestamp"] = vid_timestamps_df["timestamp (s)"].values[:-1]
    moseq_df.shape

    # Keypoint data
    keypoint_path = (
        animal_dir / "ir_clippedDLC_resnet50_KeypointMoSeqDLCOct18shuffle1_50000.csv"
    )
    keypoint_df = pd.read_csv(keypoint_path, header=[1, 2])
    temp_df = pd.read_csv(
        keypoint_path,
    )
    scorer = temp_df.columns[1]
    del temp_df

    return (
        metadata_df,
        video_paths,
        keypoint_df,
        vid_timestamps_df,
        moseq_df,
        scorer,
    )


def main():
    parser = argparse.ArgumentParser(description="Convert SAFRI data to NWB format.")
    parser.add_argument("animal_dir", type=str, help="Path to the animal directory")
    parser.add_argument(
        "output_dir", type=str, help="Directory to save the output NWB file"
    )
    args = parser.parse_args()

    animal_dir = Path(args.animal_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read in the data
    (
        metadata_df,
        video_paths,
        keypoint_df,
        vid_timestamps_df,
        moseq_df,
        scorer,
    ) = read_data(animal_dir)

    # Make file and metadata objects
    nwbfile = pynwb.NWBFile(
        session_description=metadata_df.SessionName.values[0],
        experimenter="Kind Lab",
        lab="Kind Lab",
        institution="University of Edinburgh",
        session_start_time=datetime.fromtimestamp(
            metadata_df.StartTime.values[0], pytz.utc
        ),
        timestamps_reference_time=datetime.fromtimestamp(0, pytz.utc),
        identifier=str(uuid.uuid1()),
        session_id=metadata_df.SessionName.values[0],
        experiment_description="",
        source_script="sfari_test_convert",
        source_script_file_name="convert.py",
    )

    age = metadata_df.SubjectName.values[0].split("_")[0]
    iso_age = f"P{age.strip('weeks')}W"
    genotype = metadata_df.SessionName.values[0].split("-")[0]
    subject_id = "sfari" + metadata_df.SubjectName.values[0].split("_")[-1]

    nwbfile.subject = pynwb.file.Subject(
        age=iso_age,
        genotype=genotype,
        sex="U",
        species="Rattus norvegicus",  # TODO ??
        subject_id=subject_id,
    )

    camera = nwbfile.add_device(
        CameraDevice(
            name="placeholder_camera_device 1",
            meters_per_pixel=-1.0,  # TODO ??
            manufacturer="Microsoft",
            model="Azure Kinect DK",
            lens="",
            camera_name="kinect 01",
        )
    )

    # Add Video
    # make processing module for video files
    nwbfile.create_processing_module(
        name="video_files", description="Contains all associated video files data"
    )
    # make a behavioral Event object to hold videos
    video = pynwb.behavior.BehavioralEvents(name="video")
    for video_path in video_paths:
        video.add_timeseries(
            pynwb.image.ImageSeries(
                device=camera,
                name=video_path.stem,
                timestamps=vid_timestamps_df["timestamp (s)"].values,
                external_file=[video_path],
                format="external",
                starting_frame=[0],
                description="",
            )
        )
    nwbfile.processing["video_files"].add(video)

    # DLC Pose Estimates
    nodes = np.unique([x[0] for x in keypoint_df.columns[1:]])
    edges = np.array(
        [
            [0, 1],
            [0, 5],
            [0, 6],
            [1, 5],
            [1, 6],
            [5, 12],
            [4, 12],
            [4, 11],
            [11, 7],
            [7, 8],
            [8, 9],
            [9, 10],
            [11, 2],
            [11, 3],
        ]
    )  # TODO
    skeleton = Skeleton(
        name="sfari_skeleton",
        nodes=nodes,
        edges=edges,
        subject=nwbfile.subject,
    )
    part_series = []
    for body_part in nodes:
        part_series.append(
            PoseEstimationSeries(
                name=body_part,
                description=body_part,
                data=keypoint_df[body_part][["x", "y"]].values,
                unit="pixels",
                reference_frame="",
                timestamps=vid_timestamps_df[
                    "timestamp (s)"
                ].values,  # link to timestamps of front_left_paw so we don't have to duplicate them
                confidence=keypoint_df[body_part]["likelihood"].values,
                confidence_definition="likelihood",
            )
        )

    pose_estimation = PoseEstimation(
        name="DLC_keypoints",
        pose_estimation_series=part_series,
        description="Estimated positions of keypoints using DeepLabCut.",
        original_videos=[animal_dir / "ir_clipped.avi"],
        labeled_videos=[[x for x in video_paths if "labeled" in x.stem][0]],
        dimensions=np.array(
            [[640, 480]], dtype="uint16"
        ),  # pixel dimensions of the video
        devices=[camera],
        scorer=scorer,
        source_software="DeepLabCut",
        source_software_version="",
        skeleton=skeleton,  # link to the skeleton object
    )
    behavior_pm = nwbfile.create_processing_module(
        name="behavior",
        description="processed behavioral data",
    )

    behavior_pm.add(skeleton)
    behavior_pm.add(pose_estimation)

    # Moseq syllables
    moseq_obj = TimeSeries(
        name="moseq syllables",
        data=moseq_df["syllable"].values,
        timestamps=moseq_df["timestamp"].values,
        unit="syllable label",
    )
    behavior_pm.add(moseq_obj)

    # Position
    position = pynwb.behavior.Position(name="position")
    nwbfile.processing["behavior"].add(position)
    position.create_spatial_series(
        name="moseq_centroid",
        description=", ".join(["xloc", "yloc"]),
        data=moseq_df[["centroid_x", "centroid_y"]].values,
        conversion=1.0,
        reference_frame="Upper left corner of video frame",
        timestamps=moseq_df["timestamp"].values,
    )

    # save the file
    output_path = output_dir / f"{genotype}{subject_id.split('sfari')[1]}.nwb"
    with pynwb.NWBHDF5IO(output_path, "w") as io:
        io.write(nwbfile)


if __name__ == "__main__":
    main()
