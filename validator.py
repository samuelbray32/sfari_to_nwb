import pynwb
import argparse
import nwbinspector
from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton


def main():
    parser = argparse.ArgumentParser(description="Validate Safri nwb conversion.")
    parser.add_argument("path", type=str, help="Path to the nwb file")
    args = parser.parse_args()
    path = args.path

    no_issues = True
    # pynwb validation
    if len(errors := pynwb.validate(paths=[path])):
        print("Validation errors found in the NWB file:")
        for error in errors:
            print(error)
        no_issues = False

    # dandi validation
    messages = list(
        nwbinspector.inspect_nwbfile(
            nwbfile_path=path,
            config=nwbinspector.load_config("dandi"),
            skip_validate=True,
        )
    )
    flagged_error_levels = [
        nwbinspector.Importance.ERROR,
        nwbinspector.Importance.CRITICAL,
    ]
    critical_errors = list(
        filter(lambda x: x.importance in flagged_error_levels, messages)
    )
    if critical_errors:
        print(
            f"NWB Inspector found the following {len(critical_errors)} critical errors:"
        )
        formatted_critical_errors = nwbinspector.format_messages(
            messages=critical_errors
        )
        nwbinspector.print_to_console(formatted_messages=formatted_critical_errors)
        no_issues = False

    # check for content
    io = pynwb.NWBHDF5IO(path, "r")
    nwbfile = io.read()
    for x in ["DLC_keypoints", "moseq syllables", "position", "sfari_skeleton"]:
        if x not in nwbfile.processing["behavior"].data_interfaces:
            print(f"Missing {x} in behavior processing module")
            no_issues = False

    pose = nwbfile.processing["behavior"]["DLC_keypoints"]
    if not isinstance(pose, PoseEstimation):
        print("DLC_keypoints is not a PoseEstimation object")
        no_issues = False
    expected_parts = [
        "earl",
        "earr",
        "hipl",
        "hipr",
        "mid_spine",
        "neck",
        "nose",
        "tail1",
        "tail2",
        "tail3",
        "tailend",
        "tailstart",
        "top_spine",
    ]
    for part in expected_parts:
        if part not in pose.pose_estimation_series:
            print(f"Missing {part} in pose parts")
            no_issues = False
        else:
            if not isinstance(
                pose_part := pose.pose_estimation_series[part], PoseEstimationSeries
            ):
                print(f"{part} is not a PosePart object")
                no_issues = False
            if len(pose_part.data) == 0:
                print(f"{part} has no data")
                no_issues = False

    if not isinstance(
        syllables := nwbfile.processing["behavior"]["moseq syllables"], pynwb.TimeSeries
    ):
        print("moseq syllables is not a TimeSeries object")
        no_issues = False
    if len(syllables.data) == 0:
        print("moseq syllables has no data")
        no_issues = False

    if not isinstance(
        nwbfile.processing["behavior"]["position"], pynwb.behavior.Position
    ):
        print("position is not a SpatialSeries object")
        no_issues = False

    if not isinstance(nwbfile.processing["behavior"]["sfari_skeleton"], Skeleton):
        print("sfari_skeleton is not a Skeleton object")
        no_issues = False

    if no_issues:
        print("No issues found in the NWB file.")


if __name__ == "__main__":
    main()
