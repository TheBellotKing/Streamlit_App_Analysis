import pandas as pd
import streamlit as st
from glob import glob
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
import pymedphys
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

st.title('Compare Continuous with Step by Step PDD')

# files = glob("/home/blue/Downloads/PDD_*.csv")
#
# bpfile = st.selectbox("Select the Blue Physics file", files)
#
# filenow = open(bpfile)
#
# lines = filenow.readlines()
#
# filenow.close()

file_bp = st.file_uploader("Choose a PDD from BP", type="csv", key='file_bp')

if file_bp:
    #Find rows to skip
    for number, line in enumerate(file_bp):
        if line.startswith(b'Number'):
            break
    file_bp.seek(0)

    #create the data frame
    df_bp = pd.read_csv(file_bp, skiprows=number)

    file_bp.seek(0)
    #Find depth of PDD
    for line_depth in file_bp:
        if line_depth.startswith(b'Started a depth:'):
            break
    depth = float(line_depth[17:-3])
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

    #Calculate real PDD data frame
    dfp = df_bp.loc[(df_bp.Time > start_time) & (df_bp.Time < end_time), ['Time', 'ch0z', 'ch1z']]
    with expander_options:
        ACR = st.number_input('ACR', value=used_ACR, format="%0.3f")
        pulse_range = st.selectbox('Pulse range %', options=[0, 5, 10, 15, 20, 25, 30], index=6, label_visibility='visible')

    # Calculate doses and pulses
    dfp['Dose'] = dfp.ch0z - dfp.ch1z * ACR
    dfp = refind_pulses(dfp, maximum_np, pulse_range)

    #Calculate PDD with coordinates
    dfp = calc_coordinates(dfp,start_point, end_point, start_time, end_time)

    number_pulses_mm = dfp[dfp.single_pulse].shape[0] // abs(end_point[2] - start_point[2])
    number_measurements_mm = dfp.shape[0] // abs(end_point[2] - start_point[2])

    #Data Frame of only the pulses
    #dfpdd = dfp[(dfp.single_pulse) & (dfp.complete_Dose > 1)].copy()
    dfpdd = dfp[dfp.single_pulse].copy()
    std_detector = dfp.loc[~dfp.pulse, 'Dose'].std()

    with expander_options:
        smoothfactor = st.slider('Rolling Window', value = 200, min_value = 0, max_value = 1000, step = 25)
        cols_checkboxes_options = st.columns([3, 3, 3], vertical_alignment="top")
        with cols_checkboxes_options[0]:
            show_rawdata = st.checkbox('Show Rawdata')
        with cols_checkboxes_options[1]:
            show_dmax = st.checkbox('Show Dmax')

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
        fig1 = px.scatter(dfpdd, x='Z', y='PDD', title="PDD")
        fig1.update_traces(name='BP PDD',marker=dict(opacity=1), showlegend=True)
        if show_dmax:
            idx_max = dfpdd.index[dfpdd['PDD'] >= dfpdd['PDD'].max()].max()
            maximum_x = dfpdd.loc[idx_max, 'Z']
            # Add annotation
            fig1.update_layout(
                annotations=[
                    dict(
                        x=maximum_x,  # X-coordinate of the arrow
                        y=100,  # Y-coordinate of the arrow
                        xref="x",  # Reference to the x-axis
                        yref="y",  # Reference to the y-axis
                        text=f"Dmax {maximum_x:.2f} mm",  # Text to display
                        showarrow=True,  # Show arrow
                        arrowhead=2,  # Style of the arrowhead
                        ax=60,  # X-offset of the arrow's tail
                        ay=-40,  # Y-offset of the arrow's tail
                        font=dict(
                            size=12,
                            color="lightblue"
                        ),
                        arrowcolor="lightblue"
                    )
                ]
            )

        fig1.add_traces([
            go.Scatter(
                x=dfpdd.Z,
                y=dfpdd.errorplus,
                mode='lines',
                line=dict(width=0),
                showlegend=False
             ),
            go.Scatter(
                x=dfpdd.Z,
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
                x=dfpdd.Z,
                y=dfpdd.noise_plus,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                x=dfpdd.Z,
                y=dfpdd.noise_minus,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,100,250,0.2)',
                showlegend=True,
                name='Radiation Noise'
            )
        ])
        fig1.update_yaxes(title = 'PDD (%)')
    else:
        fig1 = px.scatter(dfpdd, x='Z', y='PDDraw')
        fig1.update_yaxes(title = 'Charge proportional to dose (nC)')
    fig1.update_traces(marker=dict(size=4))
    fig1.add_vline(x=0,  line_color='black')
    fig1.update_xaxes(title = 'Z (mm)')

    if show_rawdata:
        with expander_options:
            number_pulses_maximum = st.slider("Number of measurements used for maximum:", min_value=1, max_value=100, value=50)
        maximum_rawdata_dose = dfpdd['complete_dose'].nlargest(number_pulses_maximum).mean()
        dfpdd['rawdata_normalized'] = dfpdd.complete_dose / maximum_rawdata_dose * 100

        fig1.add_scatter(
            x=dfpdd['Z'],
            y=dfpdd['rawdata_normalized'],
            mode='markers',
            name='Rawdata Normalized',
            marker=dict(color='red', opacity=0.2, size=2)
        )

    ext_file = st.file_uploader("Choose step by step PDD to compare", key='step_PDD_bp')

    if ext_file and smoothfactor != 0:
        # Find rows to skip
        for number, line in enumerate(ext_file):
            if line.startswith(b'Number'):
                break
        ext_file.seek(0)

        # create the data frame
        df_bp = pd.read_csv(ext_file, skiprows=number)

        ext_file.seek(0)

        df_bp_op = df_bp.loc[df_bp.single_pulse]
        df_bp_gr = df_bp_op.groupby('Z').agg({'complete_dose': 'mean'}).reset_index()
        df_bp_gr['pdd'] = df_bp_gr.complete_dose / df_bp_gr.complete_dose.max() * 100

        fig1.add_scatter(
            x=df_bp_gr['Z'],
            y=df_bp_gr['pdd'],
            mode='markers',
            name=f'Step by step PDD',
            marker=dict(color='green')
        )

        with expander_options:
            with cols_checkboxes_options[2]:
                show_dmax_steps = st.checkbox("Show Dmax for Steps by Steps PDD")

        if show_dmax_steps:
            idx_max = df_bp_gr.index[df_bp_gr['pdd'] >= df_bp_gr['pdd'].max()].max()
            maximum_x = df_bp_gr.loc[idx_max, 'Z']
            # Add annotation
            fig1.add_annotation(
                x=maximum_x,  # X-coordinate of the arrow
                y=100,  # Y-coordinate of the arrow
                xref="x",  # Reference to the x-axis
                yref="y",  # Reference to the y-axis
                text=f"Steps by steps Dmax {maximum_x:.2f} mm",  # Text to display
                showarrow=True,  # Show arrow
                arrowhead=2,  # Style of the arrowhead
                ax=50,  # X-offset of the arrow's tail
                ay=-10,  # Y-offset of the arrow's tail
                font=dict(
                    size=12,
                    color="lightgreen"
                ),
                arrowcolor="lightgreen",
                xanchor="left",  # Align the text left to its position
                yanchor="middle"
            )

        expander_gamma = st.expander("Gamma options", expanded=False)

        with expander_gamma:
            show_gamma = st.checkbox("Show Gamma")

        if show_gamma:

            positions_ref = (np.array(df_bp_gr['Z']),)
            doses_ref = np.array(df_bp_gr['pdd'])

            positions_comp = (np.array(dfpdd['Z']),)
            doses_comp = np.array(dfpdd['PDD'])

            with expander_gamma:
                # Gamma calculation parameters
                distance_criteria_mm = st.slider("Distance Threshold (mm)", min_value=0.0, max_value=3.0, value=1.0, step=0.25)
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
