# -*- coding: utf-8 -*-
"""
Lumped Models for Hydrology Streamlit Application


Author:
    Saul Arciniega Esparza | sarciniegae@comunidad.unam.mx
    Ph.D. on Engineering, Water Resources
    Associate Proffesor at Faculty of Engineering, UNAM
"""

#%% Import libraries
import os
import pandas as pd
import numpy as np
from PIL import Image

from plotly.subplots import make_subplots
import plotly.graph_objs as go

import streamlit as st
import lumod
from lumod import tools

root = os.path.abspath(os.path.dirname(__file__))

#%% Main functions
@st.cache_data()
def load_file(filename):
    if filename is not None:
        columns = ["prec", "tmin", "tmean", "tmax"]
        forcings = pd.read_csv(filename, index_col=[0], parse_dates=[0])
        area = 100.
        lat = 0.
        if all([col in forcings for col in columns]):
            flag = 1
        else:
            flag = 0
            info, forcings = lumod.load_example(2)
            area = float(info.area)
            lat = float(info.lat)
    else:
        flag = -1
        info, forcings = lumod.load_example(2)
        area = float(info.area)
        lat = float(info.lat)
    return forcings, area, lat, flag


@st.cache_data(allow_output_mutation=True)
def load_model(name):
    if name == "MILC":
        model = lumod.models.MILC()
    elif name == "GR4J":
        model = lumod.models.GR4J()
    elif name == "HYMOD":
        model = lumod.models.HYMOD()
    elif name == "HBV":
        model = lumod.models.HBV()  
    else:
        model = None
    return model


def load_options(name, area, lat):
    if name == "MILC":
        parameters = dict(
            area = st.sidebar.number_input('Catchment area (km2)', value=area, min_value=1.0),
            lat = st.sidebar.number_input('Catchment centroid latitude (degress)', value=lat, min_value=-90.0, max_value=90.0),
            w0 = st.sidebar.number_input('Initial water content as a fraction of wmax (-) [w0]', value=0.5, min_value=0., max_value=1.),
            wmax = st.sidebar.number_input('Maximum water content (mm) [wmax]', value=1000., min_value=0.),
            gamma = st.sidebar.number_input('Routing coefficient for HUI  (-) [gamma]', value=5., min_value=0.01),
            kc = st.sidebar.number_input('PET parameter (-) [kc]', value=1., min_value=0.01),
            alpha = st.sidebar.number_input('Runoff exponent (-) [alpha]', value=2., min_value=0.01),
            m = st.sidebar.number_input('Drainage exponent (-) [m]', value=10., min_value=0.01),
            ks = st.sidebar.number_input('Satured hydraulic conductivity (mm/d) [ks]', value=100., min_value=0.01),
            nu = st.sidebar.number_input('Fraction of drainage vs interflow (-) [nu]', value=0.5, min_value=0., max_value=1.),
        )
    elif name == "GR4J":
        parameters = dict(
            area = st.sidebar.number_input('Catchment area (km2)', value=area, min_value=1.0),
            lat = st.sidebar.number_input('Catchment centroid latitude (degress)', value=lat, min_value=-90.0, max_value=90.0),
            ps0 = st.sidebar.number_input('Initial production storage (ps/x1)', value=1.0, min_value=0., max_value=1.),
            rs0 = st.sidebar.number_input('Initial routing storage (rs/x1)', value=0.5, min_value=0., max_value=1.),
            x1 = st.sidebar.number_input('Maximum production capacity (mm) [x1]', value=500., min_value=0.),
            x2 = st.sidebar.number_input('Water exchange coefficient (mm) [x2]', value=3.),
            x3 = st.sidebar.number_input('Routing maximum capacity (mm) [x3]', value=200., min_value=0.),
            x4 = st.sidebar.number_input('Unit hydrograph time base (days) [x4]', value=5., min_value=0.),   
        )
    elif name == "HYMOD":
        parameters = dict(
            area = st.sidebar.number_input('Catchment area (km2)', value=area, min_value=1.0),
            lat = st.sidebar.number_input('Catchment centroid latitude (degress)', value=lat, min_value=-90.0, max_value=90.0),
            w0 = st.sidebar.number_input('Initial soil water content as a fraction of wmax (-) [w0]', value=0.5, min_value=0., max_value=1.),
            wmax = st.sidebar.number_input('Maximum water content (mm) [wmax]', value=800.0, min_value=0.),
            wq0 = st.sidebar.number_input('Mean initial water in quick reservoirs (mm) [wq0]', value=0.0, min_value=0.),
            ws0 = st.sidebar.number_input('Initial water in slow reservoir (mm) [ws0]', value=0., min_value=0.),
            alpha = st.sidebar.number_input('Quick-slow split parameter (-) [alpha]', value=0.3, min_value=0., max_value=1.0),
            beta = st.sidebar.number_input('Distribution function shape parameter (-) [beta]', value=1.0, min_value=0., max_value=2.0),
            cexp = st.sidebar.number_input('Exponent on ratio of maximum storage capacities (-) [cexp]', value=0.7, min_value=0., max_value=2.0),
            nres = st.sidebar.number_input('Number of quickflow routing reservoirs [nres]', value=3, min_value=1, step=1),
            ks = st.sidebar.number_input('Residence time of the slow release reservoir (1/days) [ks]', value=0.05, min_value=0.),
            kq = st.sidebar.number_input('Residence time of the quick release reservoirs (1/days) [kq]', value=0.3, min_value=0.),
            kmax = st.sidebar.number_input('Upper limit of ET resistance parameter (-) [kmax]', value=0.9, min_value=0.),
            llet = st.sidebar.number_input('Lower limit of ET resistance parameter (-) [llet]', value=0.2, min_value=0., max_value=1.0),
        )
    elif name == "HBV":
        parameters = dict(
            area = st.sidebar.number_input('Catchment area (km2)', value=area, min_value=1.0),
            lat = st.sidebar.number_input('Catchment centroid latitude (degress)', value=lat, min_value=-90.0, max_value=90.0),
            maxbas = st.sidebar.number_input('Weighting parameter used for triangular unit hydrograph (days) [maxbas]', value=3, min_value=1, step=1),
            tthres = st.sidebar.number_input('Threshold temperature for snow melt initiation (°C) [tthres]', value=5.0),
            dd = st.sidebar.number_input('Degree-day factor for snow accumulation (mm/°C.d) [dd]', value=2., min_value=0.),
            cevp = st.sidebar.number_input('PET parameter that depends of land use (mm/day.°C) [cevp]', value=2., min_value=0.),
            cevpam = st.sidebar.number_input('Amplitude of sinus function for PET (-) [cevpam]', value=1., min_value=0.),
            cevpph = st.sidebar.number_input('Phase of sinus function that corrects pet (days) [cevpph]', value=0, min_value=0, max_value=365, step=1),
            beta = st.sidebar.number_input('Shape coefficient for runoff (-) [beta]', value=2., min_value=0.),
            fc = st.sidebar.number_input('Maximum soil storage capacity (mm) [fc]', value=500., min_value=0.),
            pwp = st.sidebar.number_input('Soil permanente wilting point as a fraction of fc (-) [pwp]', value=0.8, min_value=0., max_value=1.0),
            k0 = st.sidebar.number_input('Recession coefficient of surface flow (1/d) [k0]', value=0.5, min_value=0.),
            k1 = st.sidebar.number_input('Recession coefficient of interflow (1/d) [k1]', value=0.1, min_value=0.),
            k2 = st.sidebar.number_input('Recession coefficient of baseflow (1/d) [kq]', value=0.01, min_value=0.),
            lthres = st.sidebar.number_input('Threshold water level for generating surface flow (mm) [lthres]', value=50., min_value=0.),
            snow0 = st.sidebar.number_input('Initial snow equivalent thickness (mm) [snow0]', value=0., min_value=0.),
            s0 = st.sidebar.number_input('Initial soil moisture storage (s/fc) [s0]', value=0.5, min_value=0., max_value=1.0),
            w01 = st.sidebar.number_input('Upper reservoir storage (mm) [w01]', value=20., min_value=0.),
            w02 = st.sidebar.number_input('Lower reservoir storage (mm) [w02]', value=100., min_value=0.),
        )
    else:
        parameters = {}
    return parameters


@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')


# Create app
st.set_page_config(page_title="LuMod-app", layout="wide")
st.sidebar.title("Lumped Hydrological Models (LuMod)")
st.sidebar.markdown("## Model Options")

fname = st.sidebar.file_uploader(
    label="Read forcings (.csv)",
    help="""File must contain records at daily timestep with the following columns' name: 
    dates (example: 2000-01-01), prec (precipitation in mm), tmin, tmean, tmax (temperatures in degrees), qt (optional streamflow column in m3/s)."""
)
forcings, area, lat, flag = load_file(fname)
if flag == 1:
    st.sidebar.success('Forcings were correctly loaded!')
if flag == 0:
    st.sidebar.error('Error loading your data!')

model_name = st.sidebar.selectbox("Choose a model", ("", "MILC", "GR4J", "HYMOD", "HBV"))
model = load_model(model_name)
parameters = load_options(model_name, area, lat)

st.sidebar.markdown("### About")
st.sidebar.info("""Developed by [Dr. Saul Arciniega Esparza](https://www.researchgate.net/profile/Saul-Arciniega-Esparza) from the Hydrogeology Group at the Faculty of Engineering, UNAM;
and [Christian Birkel](https://www.researchgate.net/profile/Christian-Birkel), Full Professor and Researcher at the Department of Geography at University of Costa Rica, and leader of the
Observatory of Water and Global Change (OACG).

This is a free application for learning and academic purposes, and can't be used for commercial purposes.
See the complete LuMod [user guide](https://zaul_ae.gitlab.io/lumod-docs/) or the [GitLab repository](https://gitlab.com/Zaul_AE/lumod)""")

st.sidebar.image(Image.open(os.path.join(root, "img", "logo_01.png")), width=250)

if model is not None:
    simul = model.run(forcings, **parameters)
    simul = simul.round(4)

    st.markdown(f"## Hydrological results for model {model_name}")
    
    # Precipitation plot
    dplot1 = [
            go.Scatter(
                x=forcings.index,
                y=forcings.loc[:, "prec"],
                mode="lines",
                name="Prec",
                line=dict(color='#2471a3', width=1.5)
            )
    ]

    # Streamflow plot
    dplot2 = []
    dplot3 = []
    dplot4 = []

    if "qt" in forcings:
        dplot2.append(
            go.Scatter(
                x=forcings.index,
                y=forcings.loc[:, "qt"],
                mode="lines",
                name="Qt obs",
                line=dict(color='rgb(67,67,67)', width=2)
            )
        )
        fdc_obs = tools.time_series.flow_duration_curve(forcings.loc[:, "qt"])
        dplot3.append(
            go.Scatter(
                x=fdc_obs.index,
                y=fdc_obs,
                mode="lines",
                name="Qt obs",
                line=dict(color="rgb(67,67,67)", width=2)
            )
        )
        qtm_obs = forcings.loc[:, "qt"].resample("1M").mean()
        qtm_obs = qtm_obs.groupby(qtm_obs.index.month).mean()
        dplot4.append(
            go.Scatter(
                x=qtm_obs.index,
                y=qtm_obs,
                mode="lines",
                name="Qt obs",
                line=dict(color="rgb(67,67,67)", width=2)
            )
        )

    dplot2.append(go.Scatter(
        x=simul.index,
        y=simul.loc[:, "qt"],
        mode="lines",
        name="Qt sim",
        line=dict(color="#a93226", width=1.5)
        ))
    fdc_sim = tools.time_series.flow_duration_curve(simul.loc[:, "qt"])
    dplot3.append(
        go.Scatter(
            x=fdc_sim.index,
            y=fdc_sim,
            mode="lines",
            name="Qt sim",
            line=dict(color="#a93226", width=2)
        )
    )
    qtm_sim = simul.loc[:, "qt"].resample("1M").mean()
    qtm_sim = qtm_sim.groupby(qtm_sim.index.month).mean()
    dplot4.append(
        go.Scatter(
            x=qtm_sim.index,
            y=qtm_sim,
            mode="lines",
            name="Qt sim",
            line=dict(color="#a93226", width=2)
        )
    )
    
    # Plot timeseries
    fig = make_subplots(rows=2, shared_xaxes=True)
    fig.add_trace(dplot1[0])
    if len(dplot2) == 2:
        fig.add_traces(dplot2, rows=[2,2], cols=[1, 1])
    else:
        fig.add_traces(dplot2, rows=[2], cols=[1])
    fig.update_layout(
        width=1500,
        height=600,
        legend=dict(yanchor="top", xanchor="left", x=0.01),
        hovermode="x"
    )
    fig.layout.yaxis1.update({'title': 'Precipitation (mm)'})
    fig.layout.yaxis2.update({'title': 'Streamflow (m3/s)'})
    st.plotly_chart(fig, use_container_width=True)

    # Plot validation
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(data=dplot3)
        fig.update_layout(
            title="Flow duration curve",
            xaxis_title="Excedence",
            yaxis_title="Streamflow (m3/s)"
        )
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(data=dplot4)
        fig.update_layout(
            title="Mean Monthly Qt",
            xaxis_title="Month",
            yaxis_title="Streamflow (m3/s/month)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Metrics
    if "qt" in forcings:
        st.markdown("**Model metrics**")
        col1, col2, col3 = st.columns(3)
        col1.metric("KGE", "{:.2f}".format(tools.metrics.kling_gupta_efficiency(forcings.loc[:, "qt"], simul.loc[:, "qt"])))
        col2.metric("NSE", "{:.2f}".format(tools.metrics.nash_sutcliffe_efficiency(forcings.loc[:, "qt"], simul.loc[:, "qt"])))
        col3.metric("MAE", "{:.2f} m3/s".format(tools.metrics.mean_absolute_error(forcings.loc[:, "qt"], simul.loc[:, "qt"])))

    # Export results
    st.markdown("**Donwload Area**")
    output = convert_df(simul)
    st.download_button(
        label="Download results (csv)",
        data=output,
        file_name=f"{model_name}_simulation.csv",
        mime='text/csv',
    )

else:

    st.title("Lumped Hydrological Models (LuMod)")

    st.markdown("""[**LuMod**](https://zaul_ae.gitlab.io/lumod-docs/) is an easy to use set of Lumped conceptual Models for hydrological simulation using the Python language.
    This app is a GUI for using different models incorporated in the LuMod library.""")
    
    st.subheader("Instructions")
    st.markdown("Load your data and choose a model in the left sidebar. Next, change the parameters to improve the results.")
    st.text("")
    st.markdown("If you want to use your own data for modeling, you can download the sample file by clicking the button below.")
    st.markdown("""File must contain records at daily timestep with the following columns' name: 
    dates (example: 2000-01-01), prec (precipitation in mm), tmin, tmean, tmax (temperatures in degrees), qt (optional streamflow column in m3/s).""")
    output = convert_df(lumod.load_example(2)[1])
    st.download_button(
        label="Download forcings example file",
        data=output,
        file_name="Example_Forcings.csv",
        mime='csv',
    ) 

    st.subheader("How to Cite")
    st.markdown("Coming soon...")

    st.subheader("Acknowledgments")
    
    st.markdown("""The [National Council of Science and Technology (CONACYT)](https://conacyt.mx/),
    the [Leverhulme Trust](https://www.leverhulme.ac.uk/) and the [German Academic Exchange Service (DAAD)](https://www.daad.de/en/)
    are thanked for partial funding of this work.""")

    st.image(Image.open(os.path.join(root, "img", "logo_02.png")), width=700)
    


