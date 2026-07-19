#!/usr/bin/env bash
# source /home/astribot/workspace/lcy/RoboHub/reference/astribot_sdk/env_conda.sh

# ===========================================
#   Astribot SDK Environment Setup for Conda
# ===========================================

# 1. Resolve SDK root
SDK_ROOT="/home/astribot/workspace/lcy/RoboHub/reference/astribot_sdk"

echo "[env_conda.sh] SDK_ROOT = $SDK_ROOT"

# 2. Check conda environment
if [ -z "${CONDA_PREFIX}" ]; then
    echo "[env_conda.sh][WARN] Conda environment is not active."
    echo "[env_conda.sh][WARN] Please run: conda activate astribot"
else
    echo "[env_conda.sh] CONDA_PREFIX = ${CONDA_PREFIX}"
fi

# 3. Make sure conda python is preferred
if [ -n "${CONDA_PREFIX}" ]; then
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi

# 4. Python path configuration
#    Add SDK Python modules and ROS2-generated message packages.
export PYTHONPATH="${SDK_ROOT}/third_party/astribot_ros_middleware_py:${PYTHONPATH}"
export PYTHONPATH="${SDK_ROOT}:${PYTHONPATH}"

echo "[env_conda.sh] PYTHONPATH configured."

# 5. Source ROS2 Humble
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "[env_conda.sh] ROS2 Humble sourced."
else
    echo "[env_conda.sh][WARN] /opt/ros/humble/setup.bash not found."
fi

# 6. Source SDK local setup files
if [ -f "${SDK_ROOT}/third_party/software/setup.bash" ]; then
    source "${SDK_ROOT}/third_party/software/setup.bash"
else
    echo "[env_conda.sh][WARN] ${SDK_ROOT}/third_party/software/setup.bash not found."
fi

if [ -f "${SDK_ROOT}/astribot_msgs/share/astribot_msgs/local_setup.bash" ]; then
    source "${SDK_ROOT}/astribot_msgs/share/astribot_msgs/local_setup.bash"
else
    echo "[env_conda.sh][WARN] astribot_msgs local_setup.bash not found."
fi

if [ -f "${SDK_ROOT}/third_party/third_pkg/local_setup.bash" ]; then
    source "${SDK_ROOT}/third_party/third_pkg/local_setup.bash"
else
    echo "[env_conda.sh][WARN] ${SDK_ROOT}/third_party/third_pkg/local_setup.bash not found."
fi

# 7. Library path configuration
#    Keep conda libraries first if conda is active, then append project libs.
if [ -n "${CONDA_PREFIX}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${SDK_ROOT}/astribot_sdk/core/common/robotics_library_py"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${SDK_ROOT}/astribot_sdk/core/common/whole_body_control/third_party"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${SDK_ROOT}/third_party/drake/lib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${SDK_ROOT}/third_party/third_pkg"

echo "[env_conda.sh] LD_LIBRARY_PATH updated."

# 8. Robot related environment variables
if [ -z "${ROBOT_TYPE}" ]; then
    export ROBOT_TYPE="S1"
    echo "[env_conda.sh] ROBOT_TYPE not set, defaulting to S1"
else
    echo "[env_conda.sh] ROBOT_TYPE detected: ${ROBOT_TYPE}"
fi

export ASTRIBOT_SDK_ROOT="$SDK_ROOT"

# 9. ROS2 network configuration
export ROS_DOMAIN_ID=25
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# 10. Fast DDS whitelist for 192.168.0.x network
LOCAL_192_IP=$(ip -4 addr show | awk '/inet 192\.168\.0\./ {print $2}' | head -n1 | cut -d/ -f1)

if [ -n "$LOCAL_192_IP" ]; then
    if [ "$LOCAL_192_IP" = "192.168.0.10" ]; then
        echo "[env_conda.sh] Detected local IP = 192.168.0.10, skip Fast DDS whitelist setup."
    else
        FASTDDS_TEMPLATE="${SDK_ROOT}/config/fastdds_whitelist_192.xml.template"
        FASTDDS_XML="${SDK_ROOT}/config/fastdds_whitelist_192.xml"

        if [ -f "$FASTDDS_TEMPLATE" ]; then
            sed "s/__LOCAL_192_IP__/${LOCAL_192_IP}/g" "$FASTDDS_TEMPLATE" > "$FASTDDS_XML"
            export FASTRTPS_DEFAULT_PROFILES_FILE="$FASTDDS_XML"
            echo "[env_conda.sh] Fast DDS whitelist configured for ${LOCAL_192_IP}"
        else
            echo "[env_conda.sh][WARN] Fast DDS template not found."
        fi
    fi
else
    echo "[env_conda.sh][WARN] No 192.168.0.x address detected."
fi

echo "[env_conda.sh] ROS_DOMAIN_ID      = $ROS_DOMAIN_ID"
echo "[env_conda.sh] RMW_IMPLEMENTATION = $RMW_IMPLEMENTATION"
echo "[env_conda.sh] Environment setup completed."