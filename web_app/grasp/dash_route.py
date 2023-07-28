from flask import  session
from grasp import app

import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd

# Dash application
app_dash = dash.Dash(__name__, server=app, url_base_pathname='/dash/')

# Dash layout
app_dash.layout = html.Div([
    dcc.Graph(id='graph'),
    dcc.Dropdown(
        id='variable-dropdown',
        options=[],
        value=''
    )
])


# Define the Dash callback
@app_dash.callback(Output('graph', 'figure'), [Input('variable-dropdown', 'value')])
def update_graph(selected_variable):
    # Retrieve the file path from the session
    file_path = session.get('uploaded_file_path')

    if file_path:
        # Process the uploaded file (e.g., read it into a DataFrame)
        df = pd.read_csv(file_path)

        fig = make_subplots(rows=1, cols=1)

        # Add traces for each Y variable
        for i, y_var in enumerate(df.columns[1:]):
            visible = True if y_var == selected_variable else False
            fig.add_trace(go.Scatter(x=df[df.columns[0]], y=df[y_var], name=y_var, visible=visible), row=1, col=1)

        fig.update_layout(updatemenus=[{
            'buttons': [{'label': y_var, 'method': 'update', 'args': [{'visible': [y_var == var for var in df.columns[1:]]}]} for y_var in df.columns[1:]],
            'direction': 'down',
            'showactive': True
        }])

        return fig

    # If no uploaded data, return an empty figure
    return go.Figure()
