# This scripot has arguments
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from basil.utils import compute_pareto_frontier, load_results

arg_parser = argparse.ArgumentParser(description='Process some inputs.')
arg_parser.add_argument('--folder',
                        type=str,
                        help='folder containing the benchmark runs')
args = arg_parser.parse_args()

st.title('Basil')
st.write(f'{args.folder}')

# Load all runs
result_families = load_results(args.folder)

# Converting the result_families to a pandas dataframe for easier manipulation
data = []
for family in result_families:
    for result in family.results:
        # Flatten the data
        flattened_result = {
            **result.system_info,
            **result.options,
            **result.parameters,
            **result.results
        }
        flattened_result["solver"] = result.solver
        flattened_result["name"] = family.name
        data.append(flattened_result)

df = pd.DataFrame(data)

# Filter for names
names = df['name'].unique().tolist()
selected_name = st.sidebar.selectbox("Select a BenchmarkFamily:",
                                     options=names)
df = df[df['name'] == selected_name]

# Get all unique columns from dataframe to allow user to select
cols = df.columns.tolist()

st.sidebar.write('Please select options for scatter plot')
group_by = st.sidebar.selectbox("Group results by:", options=cols, index=0)
x_axis = st.sidebar.selectbox("Select X-Axis:", options=cols, index=1)
y_axis = st.sidebar.selectbox("Select Y-Axis:", options=cols, index=2)
log_x = st.sidebar.checkbox("Log scale on X-axis", False)
log_y = st.sidebar.checkbox("Log scale on Y-axis", False)

pareto_curve = st.sidebar.checkbox("Display Pareto Curve", False)
x_better = st.sidebar.radio("Select the preferred direction for the X-axis",
                            options=["Higher is better", "Lower is better"],
                            index=1)
y_better = st.sidebar.radio("Select the preferred direction for the Y-axis",
                            options=["Higher is better", "Lower is better"],
                            index=1)

# Dynamic filtering based on column types
# Heading
st.sidebar.write('Please select filters for the data')
for col in df.columns:
    if np.issubdtype(df[col].dtype, np.number):
        min_val, max_val = df[col].min(), df[col].max()
        if min_val == max_val:
            # If min and max are the same, then the slider will not work
            # So we just show the value
            st.sidebar.write(f'{col} value: {min_val}')
        else:
            selected_range = st.sidebar.slider(
                f"{col} range", float(min_val), float(max_val),
                (float(min_val), float(max_val)))
            df = df[(df[col] >= selected_range[0])
                    & (df[col] <= selected_range[1])]
    elif pd.api.types.is_categorical_dtype(
            df[col].dtype) or pd.api.types.is_string_dtype(df[col].dtype):
        unique_vals = df[col].unique().tolist()
        selected_vals = st.sidebar.multiselect(f"{col} values",
                                               options=unique_vals,
                                               default=unique_vals)
        df = df[df[col].isin(selected_vals)]

# User can select runs to display
runs = df[group_by].unique().tolist()
selected_runs = st.sidebar.multiselect("Select runs to display:",
                                       options=runs,
                                       default=runs)

# Create grouped object
grouped = df[df[group_by].isin(selected_runs)].groupby(group_by)

fig, ax = plt.subplots()

# Scatter plot for each group
for name, group in grouped:
    ax.scatter(group[x_axis], group[y_axis], label=name, marker='x', alpha=0.5)

    if pareto_curve:
        pareto_points = compute_pareto_frontier(group, x_axis, y_axis,
                                                x_better, y_better)

        if pareto_points is not None:
            if x_better == "Lower is better":
                pareto_points[:, 0] = -pareto_points[:, 0]
            if y_better == "Lower is better":
                pareto_points[:, 1] = -pareto_points[:, 1]
            ax.plot(pareto_points[:, 0], pareto_points[:, 1])

if log_x:
    ax.set_xscale('log')
if log_y:
    ax.set_yscale('log')

ax.legend(loc='best')
plt.xlabel(x_axis)
plt.ylabel(y_axis)

st.pyplot(fig)
