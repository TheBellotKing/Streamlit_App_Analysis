import pandas as pd
import streamlit as st
from glob import glob
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np

from site_packages import pymedphys
from helpers import calc_coordinates

st.logo(image="images/logo.png", icon_image="images/icon.png")

def refind_pulses(df: pd.DataFrame, maximum_non_pulse: float, pulse_range: int = 5) -> pd.DataFrame:
    percentage = 1 + pulse_range / 100
    df['pulse'] = (df.ch0z > maximum_non_pulse * percentage)
    df['pulse_after'] = df.pulse.shift(-1)
    df['pulse_coincide'] = df.pulse + df.pulse_after == 2
    df['single_pulse'] = df.pulse
    df['pulse_coincide_after'] = df.pulse_coincide.shift()
    df.dropna(inplace=True)
    df.loc[df.pulse_coincide_after, 'single_pulse'] = False

    df['dose_after'] = 0.0
    df.loc[df.pulse_coincide_after, 'dose_after'] = df.Dose
    df['complete_dose'] = df.Dose + df.dose_after.shift(-1)
    df.loc[df.pulse_coincide_after, 'complete_dose'] = 0.0

    return df

st.title('Compare Continuous with Step by Step Profile')

# files = glob("/home/blue/Downloads/PDD_*.csv")
#
# bpfile = st.selectbox("Select the Blue Physics file", files)
#
# filenow = open(bpfile)
#
# lines = filenow.readlines()
#
# filenow.close()

file_bp = st.file_uploader("Choose a Profile from BP", type="csv", key='file_bp_profile')

if file_bp:
    #Find rows to skip
    for number, line in enumerate(file_bp):
        if line.startswith(b'Number'):
            break
    file_bp.seek(0)

    #create the data frame
    df_bp = pd.read_csv(file_bp, skiprows=number)

    file_bp.seek(0)

    #Find axis of Profile
    for line_profile_type in file_bp:
        if line_profile_type.startswith(b'Profile for'):
            if line_profile_type.startswith(b'Profile for X'):
                axis_used = 'X'
            else:
                axis_used = 'Y'
            break
    file_bp.seek(0)

    #Find depth of PDD
    for line_depth in file_bp:
        if line_depth.startswith(b'Depth:'):
            break
    depth = float(line_depth[7:-3])
    file_bp.seek(0)

    #Find time start PDD
    for line_start in file_bp:
        if line_start.startswith(b'Movement start:'):
            break
    start_time = float(line_start[16:-2])
    file_bp.seek(0)

    #Find time end PDD
    for line_end in file_bp:
        if line_end.startswith(b'Movement end:'):
            break
    end_time = float(line_end[14:-2])
    file_bp.seek(0)

    #Find startpoint PDD
    for line_position_search in file_bp:
        if line_position_search.startswith(b'Movement start X:'):
            x_start = float(line_position_search[18:-3])
        if line_position_search.startswith(b'Movement start Y:'):
            y_start = float(line_position_search[18:-3])
        if line_position_search.startswith(b'Movement start Z:'):
            z_start = float(line_position_search[18:-3])
        if line_position_search.startswith(b'Movement end X:'):
            x_end = float(line_position_search[15:-3])
        if line_position_search.startswith(b'Movement end Y:'):
            y_end = float(line_position_search[15:-3])
        if line_position_search.startswith(b'Movement end Z:'):
            z_end = float(line_position_search[15:-3])
            break
    start_point = [x_start,y_start,z_start]
    end_point = [x_end,y_end,z_end]
    file_bp.seek(0)

    #Find ACR used:
    for line_ACR in file_bp:
        if line_ACR.startswith(b'ACR used:'):
            break
    used_ACR = float(line_ACR[9:])
    file_bp.seek(0)

    # Find maximum non pulse used
    for line_max_np in file_bp:
        if line_max_np.startswith(b'Maximum non pulse:'):
            break
    maximum_np = float(line_max_np[19:])
    file_bp.seek(0)

    expander_options = st.expander("Analysis options", expanded=False)

    #Calculate real Profile data frame
    dfp = df_bp.loc[(df_bp.Time > start_time) & (df_bp.Time < end_time), ['Time', 'ch0z', 'ch1z']]
    with expander_options:
        ACR = st.number_input('ACR', value=used_ACR, format="%0.3f")
        pulse_range = st.selectbox('Pulse range %', options=[0, 5, 10, 15, 20, 25, 30], index=6, label_visibility='visible')

    # Calculate doses and pulses
    dfp['Dose'] = dfp.ch0z - dfp.ch1z * ACR
    dfp = refind_pulses(dfp, maximum_np, pulse_range)

    #Calculate PDD with coordinates
    dfp = calc_coordinates(dfp,start_point, end_point, start_time, end_time)

    if axis_used == 'X':
        number_pulses_mm = dfp[dfp.single_pulse].shape[0] // abs(end_point[0] - start_point[0])
        number_measurements_mm = dfp.shape[0] // abs(end_point[0] - start_point[0])
    else:
        number_pulses_mm = dfp[dfp.single_pulse].shape[0] // abs(end_point[1] - start_point[1])
        number_measurements_mm = dfp.shape[0] // abs(end_point[1] - start_point[1])

    #Data Frame of only the pulses
    #dfpdd = dfp[(dfp.single_pulse) & (dfp.complete_Dose > 1)].copy()
    dfpdd = dfp[dfp.single_pulse].copy()
    std_detector = dfp.loc[~dfp.pulse, 'Dose'].std()

    with expander_options:
        smoothfactor = st.slider('Rolling Window', value=50, min_value=0, max_value=1000, step=25)
        cols_checkboxes_options_1 = st.columns([3, 3, 3], vertical_alignment="top")
        cols_checkboxes_options_2 = st.columns([3, 3, 3], vertical_alignment="top")
        with cols_checkboxes_options_1[0]:
            show_field_size = st.checkbox("Show field size")
        with cols_checkboxes_options_1[1]:
            show_penumbra = st.checkbox("Show penumbra")
        with cols_checkboxes_options_2[0]:
            show_rawdata = st.checkbox('Show Rawdata')
        with cols_checkboxes_options_2[1]:
            if show_rawdata:
                show_field_size_rawdata = st.checkbox("Show field size for rawdata")
        with cols_checkboxes_options_2[2]:
            if show_rawdata:
                show_penumbra_rawdata = st.checkbox("Show penumbra for rawdata")

    #Smooth the dataframe
    dfpdd['smoothdose'] = dfpdd.complete_dose.rolling(smoothfactor, center=True).mean()
    dfpdd['PDDraw'] = dfpdd.complete_dose
    if smoothfactor != 0:
        dfpdd['radiation_noise'] = dfpdd.complete_dose.rolling(smoothfactor, center=True).std()
        dfpdd['myerror'] = std_detector / smoothfactor**0.5
        dfpdd['std_normalized'] = dfpdd.myerror / dfpdd.smoothdose.max() * 100
        dfpdd['noise_normalized'] = dfpdd.radiation_noise / dfpdd.smoothdose.max() * 100
        dfpdd['PDD'] = dfpdd.smoothdose / dfpdd.smoothdose.max() * 100
        dfpdd['errorplus'] = dfpdd.PDD + dfpdd.std_normalized
        dfpdd['errorminus'] = dfpdd.PDD - dfpdd.std_normalized
        dfpdd['noise_plus'] = dfpdd.PDD + dfpdd.noise_normalized
        dfpdd['noise_minus'] = dfpdd.PDD - dfpdd.noise_normalized
        fig1 = px.scatter(dfpdd, x=axis_used, y='PDD', title=f'Profile in {axis_used} axis and Depth: {depth} mm')
        fig1.update_traces(name=f'BP {axis_used}-axis Profile',marker=dict(opacity=1), showlegend=True)
        fig1.add_vline(0, line_width=1, line_dash="dash", line_color="grey", annotation_text="",
                            annotation_position="top right")
        if show_field_size:
            # Filter rows where pdd > 50
            filtered_rows = dfpdd[dfpdd['PDD'] > 50]
            # Find the lowest and highest Z value among the filtered rows
            minimum_axis = filtered_rows[axis_used].min()
            maximum_axis = filtered_rows[axis_used].max()
            text_position = (minimum_axis+maximum_axis)/2
            # Add a line with arrowheads at both ends using shapes
            fig1.add_shape(
                type="line",
                x0=minimum_axis, y0=49.9,  # Start of the line
                x1=maximum_axis, y1=49.9,  # End of the line
                line=dict(color="deepskyblue", width=1),
            )
            # Add text in the middle
            fig1.add_annotation(
                x=text_position,  # Middle of the line
                y=44,
                text=f"Field Size: {abs(minimum_axis-maximum_axis):.2f} mm",
                showarrow=False,
                font=dict(size=12, color="deepskyblue")
            )
        if show_penumbra:
            # Add penumbra zones
            # Filter rows where pdd > 50
            filtered_rows_penumbra_low = dfpdd[dfpdd['PDD'] > 20]
            filtered_rows_penumbra_high = dfpdd[dfpdd['PDD'] > 80]
            # Find the lowest and highest Z value among the filtered rows
            start_penumbra_left = filtered_rows_penumbra_low[axis_used].min()
            end_penumbra_left = filtered_rows_penumbra_high[axis_used].min()
            penumbra_left = end_penumbra_left - start_penumbra_left
            start_penumbra_right = filtered_rows_penumbra_high[axis_used].max()
            end_penumbra_right = filtered_rows_penumbra_low[axis_used].max()
            penumbra_right = end_penumbra_right - start_penumbra_right
            fig1.add_shape(
                type="rect",
                x0=start_penumbra_left,  # Start of the rectangle on the x-axis
                x1=end_penumbra_left,  # End of the rectangle on the x-axis
                y0=0,  # Start of the rectangle on the y-axis
                y1=100,  # End of the rectangle on the y-axis
                fillcolor="yellow",  # Color of the rectangle
                opacity=0.2,  # Opacity of the rectangle
                line_width=1  # No border for the rectangle
            )
            fig1.add_shape(
                type="rect",
                x0=start_penumbra_right,  # Start of the rectangle on the x-axis
                x1=end_penumbra_right,  # End of the rectangle on the x-axis
                y0=0,  # Start of the rectangle on the y-axis
                y1=100,  # End of the rectangle on the y-axis
                fillcolor="yellow",  # Color of the rectangle
                opacity=0.2,  # Opacity of the rectangle
                line_width=1  # No border for the rectangle
            )
            fig1.add_annotation(
                x=end_penumbra_left,  # X-coordinate of the arrow
                y=20,  # Y-coordinate of the arrow
                xref="x",  # Reference to the x-axis
                yref="y",  # Reference to the y-axis
                text=f"Pen.{penumbra_left:.2f}mm",  # Text to display
                showarrow=True,  # Show arrow
                arrowhead=2,  # Style of the arrowhead
                ax=60,  # X-offset of the arrow's tail
                ay=20,  # Y-offset of the arrow's tail
                font=dict(
                    size=12,
                    color="deepskyblue"
                ),
                xanchor="left",  # Align the text left to its position
                yanchor="middle",  # Align the text vertically to the middle
                arrowcolor="deepskyblue",
                opacity=1
            )
            fig1.add_annotation(
                x=start_penumbra_right,  # X-coordinate of the arrow
                y=20,  # Y-coordinate of the arrow
                xref="x",  # Reference to the x-axis
                yref="y",  # Reference to the y-axis
                text=f"Pen.{penumbra_right:.2f}mm",  # Text to display
                showarrow=True,  # Show arrow
                arrowhead=2,  # Style of the arrowhead
                ax=-60,  # X-offset of the arrow's tail
                ay=-20,  # Y-offset of the arrow's tail
                font=dict(
                    size=12,
                    color="deepskyblue"
                ),
                xanchor="right",  # Align the text left to its position
                yanchor="bottom",  # Align the text vertically to the middle
                arrowcolor="deepskyblue",
                opacity=1
            )

        fig1.add_traces([
            go.Scatter(
                x=dfpdd[axis_used],
                y=dfpdd.errorplus,
                mode='lines',
                line=dict(width=0),
                showlegend=False
             ),
            go.Scatter(
                x=dfpdd[axis_used],
                y=dfpdd.errorminus,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,250,100,0.2)',
                showlegend=True,
                name='Sensor error'
            )
        ])
        fig1.add_traces([
            go.Scatter(
                x=dfpdd[axis_used],
                y=dfpdd.noise_plus,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                x=dfpdd[axis_used],
                y=dfpdd.noise_minus,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,100,250,0.2)',
                showlegend=True,
                name='Radiation Noise'
            )
        ])
        fig1.update_yaxes(title = f'{axis_used} axis Profile (%)')
    else:
        fig1 = px.scatter(dfpdd, x=axis_used, y='PDDraw')
        fig1.update_yaxes(title = 'Charge proportional to dose (nC)')
    fig1.update_traces(marker=dict(size=4))
    # fig1.add_vline(x=0,  line_color='black')
    fig1.update_xaxes(title = f'{axis_used} (mm)')

    if show_rawdata:
        with expander_options:
            number_pulses_maximum = st.slider("Number of measurements used for maximum Rawdata:", min_value=1, max_value=100, value=50)
        maximum_rawdata_dose = dfpdd['complete_dose'].nlargest(number_pulses_maximum).mean()
        dfpdd['rawdata_normalized'] = dfpdd.complete_dose / maximum_rawdata_dose * 100

        fig1.add_scatter(
            x=dfpdd[axis_used],
            y=dfpdd['rawdata_normalized'],
            mode='markers',
            name='Rawdata Normalized',
            marker=dict(color='red', opacity=0.2, size=2)
        )
        if show_field_size_rawdata:
            # Filter rows where pdd > 50
            filtered_rows_raw = dfpdd[dfpdd['rawdata_normalized'] > 50]
            # Find the lowest and highest Z value among the filtered rows
            minimum_axis = filtered_rows_raw[axis_used].min()
            maximum_axis = filtered_rows_raw[axis_used].max()
            text_position = (minimum_axis + maximum_axis) / 2
            # Add a line with arrowheads at both ends using shapes
            fig1.add_shape(
                type="line",
                x0=minimum_axis, y0=50.1,  # Start of the line
                x1=maximum_axis, y1=50.1,  # End of the line
                line=dict(color="red", width=1),
            )

            # Add text in the middle
            fig1.add_annotation(
                x=text_position,  # Middle of the line
                y=56,
                text=f"Field Size Rawdata: {abs(minimum_axis - maximum_axis):.2f} mm",
                showarrow=False,
                font=dict(size=12, color="red")
            )

        if show_penumbra_rawdata:
            # Add penumbra zones
            filtered_rows_penumbra_low = dfpdd[dfpdd['rawdata_normalized'] > 20]
            filtered_rows_penumbra_high = dfpdd[dfpdd['rawdata_normalized'] > 80]
            # Find the lowest and highest Z value among the filtered rows
            start_penumbra_left = filtered_rows_penumbra_low[axis_used].min()
            end_penumbra_left = filtered_rows_penumbra_high[axis_used].min()
            penumbra_left = end_penumbra_left - start_penumbra_left
            start_penumbra_right = filtered_rows_penumbra_high[axis_used].max()
            end_penumbra_right = filtered_rows_penumbra_low[axis_used].max()
            penumbra_right = end_penumbra_right - start_penumbra_right
            fig1.add_shape(
                type="rect",
                x0=start_penumbra_left,  # Start of the rectangle on the x-axis
                x1=end_penumbra_left,  # End of the rectangle on the x-axis
                y0=0,  # Start of the rectangle on the y-axis
                y1=100,  # End of the rectangle on the y-axis
                fillcolor="lightcoral",  # Color of the rectangle
                opacity=0.2,  # Opacity of the rectangle
                line_width=1  # No border for the rectangle
            )
            fig1.add_shape(
                type="rect",
                x0=start_penumbra_right,  # Start of the rectangle on the x-axis
                x1=end_penumbra_right,  # End of the rectangle on the x-axis
                y0=0,  # Start of the rectangle on the y-axis
                y1=100,  # End of the rectangle on the y-axis
                fillcolor="lightcoral",  # Color of the rectangle
                opacity=0.2,  # Opacity of the rectangle
                line_width=1  # No border for the rectangle
            )
            fig1.add_annotation(
                x=end_penumbra_left,  # X-coordinate of the arrow
                y=20,  # Y-coordinate of the arrow
                xref="x",  # Reference to the x-axis
                yref="y",  # Reference to the y-axis
                text=f"Raw Pen.{penumbra_left:.2f} mm",  # Text to display
                showarrow=True,  # Show arrow
                arrowhead=2,  # Style of the arrowhead
                ax=60,  # X-offset of the arrow's tail
                ay=-10,  # Y-offset of the arrow's tail
                font=dict(
                    size=12,
                    color="lightcoral"
                ),
                xanchor="left",  # Align the text left to its position
                yanchor="middle",  # Align the text vertically to the middle
                arrowcolor="lightcoral",
                opacity=1
            )
            fig1.add_annotation(
                x=start_penumbra_right,  # X-coordinate of the arrow
                y=20,  # Y-coordinate of the arrow
                xref="x",  # Reference to the x-axis
                yref="y",  # Reference to the y-axis
                text=f"Raw Pen.{penumbra_right:.2f} mm",  # Text to display
                showarrow=True,  # Show arrow
                arrowhead=2,  # Style of the arrowhead
                ax=-60,  # X-offset of the arrow's tail
                ay=10,  # Y-offset of the arrow's tail
                font=dict(
                    size=12,
                    color="lightcoral"
                ),
                xanchor="right",  # Align the text left to its position
                yanchor="middle",  # Align the text vertically to the middle
                arrowcolor="lightcoral",
                opacity=1
            )

    ext_file = st.file_uploader("Choose step by step Profile to compare", key='step_bp')

    if ext_file and smoothfactor != 0:
        # Find rows to skip
        for number, line in enumerate(ext_file):
            if line.startswith(b'Number'):
                break
        ext_file.seek(0)

        # create the data frame
        df_bp = pd.read_csv(ext_file, skiprows=number)

        ext_file.seek(0)

        # Find axis of Profile
        for line_profile_type in ext_file:
            if line_profile_type.startswith(b'Profile for'):
                if line_profile_type.startswith(b'Profile for X'):
                    axis_used = 'X'
                else:
                    axis_used = 'Y'
                break
        ext_file.seek(0)

        df_bp_op = df_bp.loc[df_bp.single_pulse]
        df_bp_gr = df_bp_op.groupby(axis_used).agg({'complete_dose': 'mean'}).reset_index()
        df_bp_gr['profile'] = df_bp_gr.complete_dose / df_bp_gr.complete_dose.max() * 100

        fig1.add_scatter(
            x=df_bp_gr[axis_used],
            y=df_bp_gr['profile'],
            mode='markers',
            name=f'Step by step {axis_used} axis Profile',
            marker=dict(color='green')
        )

        with expander_options:
            cols_checkboxes_options_3 = st.columns([3, 3, 3], vertical_alignment="top")
            with cols_checkboxes_options_3[0]:
                show_field_size_steps = st.checkbox("Show field size for Steps")
            with cols_checkboxes_options_3[1]:
                show_penumbra_steps = st.checkbox("Show penumbra for Steps")

        if show_field_size_steps:
            # Filter rows where pdd > 50
            filtered_rows_steps = df_bp_gr[df_bp_gr['profile'] > 50]
            # Find the lowest and highest Z value among the filtered rows
            minimum_axis = filtered_rows_steps[axis_used].min()
            maximum_axis = filtered_rows_steps[axis_used].max()
            text_position = (minimum_axis + maximum_axis) / 2
            # Add a line with arrowheads at both ends using shapes
            fig1.add_shape(
                type="line",
                x0=minimum_axis, y0=50.3,  # Start of the line
                x1=maximum_axis, y1=50.3,  # End of the line
                line=dict(color="green", width=1),
            )

            # Add text in the middle
            fig1.add_annotation(
                x=text_position,  # Middle of the line
                y=62,
                text=f"Field Size Steps: {abs(minimum_axis - maximum_axis):.2f} mm",
                showarrow=False,
                font=dict(size=12, color="green")
            )

        if show_penumbra_steps:
            # Add penumbra zones
            filtered_rows_penumbra_low = df_bp_gr[df_bp_gr['profile'] > 20]
            filtered_rows_penumbra_high = df_bp_gr[df_bp_gr['profile'] > 80]
            # Find the lowest and highest Z value among the filtered rows
            start_penumbra_left = filtered_rows_penumbra_low[axis_used].min()
            end_penumbra_left = filtered_rows_penumbra_high[axis_used].min()
            penumbra_left = end_penumbra_left - start_penumbra_left
            start_penumbra_right = filtered_rows_penumbra_high[axis_used].max()
            end_penumbra_right = filtered_rows_penumbra_low[axis_used].max()
            penumbra_right = end_penumbra_right - start_penumbra_right
            fig1.add_shape(
                type="rect",
                x0=start_penumbra_left,  # Start of the rectangle on the x-axis
                x1=end_penumbra_left,  # End of the rectangle on the x-axis
                y0=0,  # Start of the rectangle on the y-axis
                y1=100,  # End of the rectangle on the y-axis
                fillcolor="green",  # Color of the rectangle
                opacity=0.2,  # Opacity of the rectangle
                line_width=1  # No border for the rectangle
            )
            fig1.add_shape(
                type="rect",
                x0=start_penumbra_right,  # Start of the rectangle on the x-axis
                x1=end_penumbra_right,  # End of the rectangle on the x-axis
                y0=0,  # Start of the rectangle on the y-axis
                y1=100,  # End of the rectangle on the y-axis
                fillcolor="green",  # Color of the rectangle
                opacity=0.2,  # Opacity of the rectangle
                line_width=1  # No border for the rectangle
            )
            fig1.add_annotation(
                x=start_penumbra_left,  # X-coordinate of the arrow
                y=20,  # Y-coordinate of the arrow
                xref="x",  # Reference to the x-axis
                yref="y",  # Reference to the y-axis
                text=f"Steps Pen.{penumbra_left:.2f} mm",  # Text to display
                showarrow=True,  # Show arrow
                arrowhead=2,  # Style of the arrowhead
                ax=-40,  # X-offset of the arrow's tail
                ay=-10,  # Y-offset of the arrow's tail
                font=dict(
                    size=12,
                    color="green"
                ),
                xanchor="right",  # Align the text left to its position
                yanchor="middle",  # Align the text vertically to the middle
                arrowcolor="green",
                opacity=1
            )
            fig1.add_annotation(
                x=end_penumbra_right,  # X-coordinate of the arrow
                y=20,  # Y-coordinate of the arrow
                xref="x",  # Reference to the x-axis
                yref="y",  # Reference to the y-axis
                text=f"Steps Pen.{penumbra_right:.2f} mm",  # Text to display
                showarrow=True,  # Show arrow
                arrowhead=2,  # Style of the arrowhead
                ax=40,  # X-offset of the arrow's tail
                ay=-10,  # Y-offset of the arrow's tail
                font=dict(
                    size=12,
                    color="green"
                ),
                xanchor="left",  # Align the text left to its position
                yanchor="middle",  # Align the text vertically to the middle
                arrowcolor="green",
                opacity=1
            )

        expander_gamma = st.expander("Gamma options", expanded=False)
        with expander_gamma:
            show_gamma = st.checkbox("Show Gamma")

        if show_gamma:
            positions_ref = (np.array(df_bp_gr[axis_used]),)
            doses_ref = np.array(df_bp_gr['profile'])

            positions_comp = (np.array(dfpdd[axis_used]),)
            doses_comp = np.array(dfpdd['PDD'])

            with expander_gamma:
                # Gamma calculation parameters
                distance_criteria_mm = st.slider("Distance Threshold (mm)", min_value=0.0, max_value=3.0, value=1.0,
                                                 step=0.25)
                dose_difference_criteria = st.slider("Dose diff %", min_value=0.0, max_value=3.0, value=1.0, step=0.25)
                local_global = st.radio('Local or Global Gamma?', ('Local', 'Global'), index=0)

            # Optional parameters
            lower_percent_dose_cutoff = 5  # Minimum dose cutoff in percentage
            interp_fraction = 10  # Fraction for interpolation

            # Gamma calculation
            gamma_map = pymedphys.gamma(
                axes_reference=positions_ref,
                dose_reference=doses_ref,
                axes_evaluation=positions_comp,
                dose_evaluation=doses_comp,
                dose_percent_threshold=dose_difference_criteria,
                distance_mm_threshold=distance_criteria_mm,
                lower_percent_dose_cutoff=lower_percent_dose_cutoff,
                interp_fraction=interp_fraction,
                local_gamma=local_global,
                random_subset=None
            )

            # Mask invalid gamma points
            valid_gamma = gamma_map[~np.isnan(gamma_map)]

            st.write("")
            st.write(f" Gamma Pass rate: {(np.mean(valid_gamma <= 1) * 100):.2f} %")

            # Scatter plot for gamma_map < 1 (lightgreen)
            fig1.add_scatter(
                x=positions_ref[0][gamma_map < 1],
                y=np.zeros(np.sum(gamma_map < 1)),
                mode='markers',
                name='Gamma < 1',
                marker=dict(color='lightgreen', size=8, opacity=0.5)
            )

            # Scatter plot for gamma_map >= 1 (lightcoral)
            fig1.add_scatter(
                x=positions_ref[0][gamma_map >= 1],
                y=np.zeros(np.sum(gamma_map >= 1)),
                mode='markers',
                name='Gamma >= 1',
                marker=dict(color='lightcoral', size=8, opacity=0.5)
            )

    st.plotly_chart(fig1)

    st.write(f'Number of measurements per mm: {int(number_measurements_mm)}')
    st.write(f'Number of pulses per mm: {int(number_pulses_mm)}')

