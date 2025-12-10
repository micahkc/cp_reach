import streamlit as st
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from io import BytesIO
import os
from streamlit_extras.stylable_container import stylable_container

import sys
sys.path.append('../')
from cp_reach.rover import reachable_set

st.set_page_config(page_title="Rover", page_icon="🚗", layout="wide")

st.markdown("<h1 style='text-align: center;'>Rover</h1>", unsafe_allow_html=True)

# Default values for rover
default_data = {
    "eps_controller": 20,
    "emi_disturbance": 20,
    "heading": 0,
    "width": 0.3,
    "COM_height": 0.06,
    "turn_radius": 8
}

# Initialize session state
if "rover_config" not in st.session_state:
    st.session_state.rover_config = default_data.copy()
if "csv_files" not in st.session_state:
    st.session_state.csv_files = []
if "folder_csv_files" not in st.session_state:
    st.session_state.folder_csv_files = []
if "img_bytes1" not in st.session_state:
    st.session_state.img_bytes1 = None
if "img_bytes2" not in st.session_state:
    st.session_state.img_bytes2 = None

# Layout columns
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Upload Rover Configuration File (JSON)")
    uploaded_file = st.file_uploader("Choose a JSON file", type=["json"])
    if uploaded_file is not None:
        try:
            file_contents = uploaded_file.read().decode("utf-8")
            loaded_json = json.loads(file_contents)
            st.session_state.rover_config.update(loaded_json)
            st.success("JSON file uploaded successfully!")
        except Exception as e:
            st.error(f"Failed to read JSON file: {e}")

    config = st.session_state.rover_config

    st.markdown("### Configure Rover Parameters")
    config_keys = {
        "eps_controller": "EPS Controller",
        "emi_disturbance": "EMI Disturbance",
        "heading": "Initial direction of Rover",
        "width": "Rover Width",
        "COM_height": "Center of Mass Height",
        "turn_radius": "Turn Radius"
    }

    for key, label in config_keys.items():
        st.session_state.rover_config[key] = st.number_input(
            label,
            value=st.session_state.rover_config.get(key, 0),
            key=key
        )

    custom_filename = st.text_input("Enter a custom file name (without extension):", "rover_config")
    config_json = json.dumps(st.session_state.rover_config, indent=4)
    st.download_button(
        label="Download Configuration",
        data=config_json,
        file_name=f"{custom_filename}.json",
        mime="application/json"
    )

    st.markdown("### Upload CSV Files with Points (x,y)")
    uploaded_files = st.file_uploader("Upload multiple CSV files", type=["csv"], accept_multiple_files=True)

    uploaded_folder = st.file_uploader("Upload all files from a folder (select all)", type=["csv"], accept_multiple_files=True, key="folder_upload")
    if uploaded_folder:
        st.markdown("**Files in uploaded folder:**")
        st.session_state.folder_csv_files.clear()
        base_label = uploaded_folder[0].name.rsplit("_", 1)[0] if uploaded_folder else "folder"
        for f in uploaded_folder:
            st.write(f.name)
            try:
                df = pd.read_csv(f)
                if "PE" in df.columns and "PN" in df.columns:
                    df = df[["PE", "PN"]].rename(columns={"PE": "x", "PN": "y"})
                elif "x_meas" in df.columns and "y_meas" in df.columns:
                    df = df[["x_meas", "y_meas"]].rename(columns={"x_meas": "x", "y_meas": "y"})
                elif df.shape[1] >= 2:
                    df = df.iloc[:, :2]
                    df.columns = ["x", "y"]
                else:
                    raise ValueError("CSV must have at least two columns")
                st.session_state.folder_csv_files.append({
                    "name": base_label,
                    "points": df.to_numpy()
                })
            except Exception as e:
                st.error(f"Error reading {f.name}: {e}")

    st.session_state.csv_files.clear()

    if uploaded_files:
        for f in uploaded_files:
            try:
                df = pd.read_csv(f)
                if "PE" in df.columns and "PN" in df.columns:
                    df = df[["PE", "PN"]].rename(columns={"PE": "x", "PN": "y"})
                elif "x_meas" in df.columns and "y_meas" in df.columns:
                    df = df[["x_meas", "y_meas"]].rename(columns={"x_meas": "x", "y_meas": "y"})
                elif df.shape[1] >= 2:
                    df = df.iloc[:, :2]
                    df.columns = ["x", "y"]
                else:
                    raise ValueError("CSV must have at least two columns")
                st.session_state.csv_files.append({
                    "name": f.name.rsplit(".", 1)[0].replace("_", " "),
                    "points": df.to_numpy(),
                    "translation": [0.0, 0.0],
                    "flip_x": False,
                    "flip_y": False,
                    "flip_xy": False,
                    "from_folder": False
                })
                st.success(f"CSV {f.name} loaded successfully!")
            except Exception as e:
                st.error(f"Error reading {f.name}: {e}")

    for idx, file_data in enumerate(st.session_state.csv_files):
        with st.expander(f"Transform: {file_data['name']}"):
            col_a, col_b = st.columns(2)

            with col_a:
                file_data["flip_x"] = st.checkbox("Flip Horizontally", value=file_data["flip_x"], key=f"flip_x_{idx}")
                file_data["flip_y"] = st.checkbox("Flip Vertically", value=file_data["flip_y"], key=f"flip_y_{idx}")
                file_data["flip_xy"] = st.checkbox("Flip XY (both)", value=file_data["flip_xy"], key=f"flip_xy_{idx}")

            with col_b:
                dx = st.number_input(f"Translate X", value=file_data["translation"][0], step=0.1, key=f"dx_{idx}")
                dy = st.number_input(f"Translate Y", value=file_data["translation"][1], step=0.1, key=f"dy_{idx}")
                file_data["translation"] = [dx, dy]

            transformed = file_data["points"].copy()
            if file_data["flip_xy"]:
                transformed *= -1
            else:
                if file_data["flip_x"]:
                    transformed[:, 0] *= -1
                if file_data["flip_y"]:
                    transformed[:, 1] *= -1
            transformed[:, 0] += file_data["translation"][0]
            transformed[:, 1] += file_data["translation"][1]

            csv_bytes = BytesIO()
            np.savetxt(csv_bytes, transformed, delimiter=",", header="x,y", comments="")
            st.download_button(
                label=f"Download Transformed CSV for {file_data['name']}",
                data=csv_bytes.getvalue(),
                file_name=f"{file_data['name'].replace(' ', '_').lower()}_transformed.csv",
                mime="text/csv",
                key=f"download_{idx}"
            )

with col2:
    def transform_points(file_data):
        points = file_data["points"].copy()
        if file_data["flip_xy"]:
            points *= -1
        else:
            if file_data["flip_x"]:
                points[:, 0] *= -1
            if file_data["flip_y"]:
                points[:, 1] *= -1
        dx, dy = file_data["translation"]
        points[:, 0] += dx
        points[:, 1] += dy
        return points

    def plot_rover_and_points(config):
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        reachable_set.emi_disturbance(config, ax1)

        if st.session_state.csv_files:
            for i, file_data in enumerate(st.session_state.csv_files):
                points = transform_points(file_data)
                ax1.plot(points[:, 0], points[:, 1], '-', alpha=1, color="Yellow", label=file_data["name"] if i == 0 else None)

        if st.session_state.folder_csv_files:
            for i, folder_data in enumerate(st.session_state.folder_csv_files):
                points = folder_data["points"]
                ax1.plot(points[:, 0], points[:, 1], '-', alpha=0.1, color="green")
                if i == 0:
                    ax1.plot([], [], alpha=0.1, color="green", label=folder_data["name"])

        ax1.legend()
        ax1.axis('equal')
        ax1.relim()
        ax1.autoscale_view()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        reachable_set.roll_over(config, ax2)
        st.pyplot(fig2)

        img_bytes1 = BytesIO()
        fig1.savefig(img_bytes1, format='png')
        img_bytes1.seek(0)
        st.session_state.img_bytes1 = img_bytes1

        img_bytes2 = BytesIO()
        fig2.savefig(img_bytes2, format='png')
        img_bytes2.seek(0)
        st.session_state.img_bytes2 = img_bytes2

    with stylable_container(
        "green",
        css_styles="""
        button {
            background-color: #3CB371;
            color: black;
            font-size: 24px;
            padding: 20px 40px;
            border-radius: 10px;
            border: none;
            width: 100%;
            cursor: pointer;
        }
        button:hover {
            background-color: #2E8B57;
        }
        """,
    ):
        analyze_button = st.button("Analyze Rover", key="analyze_rover")

    if analyze_button:
        plot_rover_and_points(st.session_state.rover_config)

    if st.session_state.img_bytes1:
        st.image(st.session_state.img_bytes1, caption="Plot 1: EMI Disturbance", use_container_width=True)
        st.download_button(
            label="Download Plot 1: EMI Disturbance",
            data=st.session_state.img_bytes1,
            file_name="rover_emi_dist.png",
            mime="image/png"
        )

    if st.session_state.img_bytes2:
        st.image(st.session_state.img_bytes2, caption="Plot 2: Roll Over", use_container_width=True)
        st.download_button(
            label="Download Plot 2: Roll Over",
            data=st.session_state.img_bytes2,
            file_name="rover_roll_over.png",
            mime="image/png"
        )
