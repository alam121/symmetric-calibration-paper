#!/bin/bash
# Run trials FIRST..LAST from run_sheet_R.csv; stop at the first outcome that differs from the prediction
# (direct, mismatched -> tilt/box abort; matched, analytic_exact -> success), any IK refusal, or any error.
R="$(cd "$(dirname "$0")" && pwd)"
unset CONDA_SHLVL
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=6 RMW_IMPLEMENTATION=rmw_fastrtps_cpp FASTRTPS_DEFAULT_PROFILES_FILE=$R/fastdds_robot.xml
for n in $(seq "$1" "$2"); do
    out=$(timeout 120 /usr/bin/python3 "$R/run_trial_frame180.py" "$n" 2>&1); rc=$?
    echo "$out" | grep -E "^run $n:" | tail -1
    mode=$(echo "$out" | head -1 | sed -E 's/.*mode (\w+)$/\1/')
    last=$(echo "$out" | grep -E "^run $n: " | tail -1)
    if [ $rc -ne 0 ]; then echo "STOP after run $n: exit $rc"; echo "$out" | tail -5; exit 1; fi
    if ! echo "$last" | grep -q "IK refusals 0"; then echo "STOP after run $n: IK refusals"; exit 1; fi
    if [ "$mode" = mismatched ] || [ "$mode" = direct ]; then
        echo "$last" | grep -qE "abort: (tool tilted|TCP left the box)" || { echo "STOP after run $n: $mode did not abort"; exit 1; }
    else
        echo "$last" | grep -q ": success;" || { echo "STOP after run $n: $mode did not succeed"; exit 1; }
    fi
    sleep 2
done
echo "DONE $1-$2"
