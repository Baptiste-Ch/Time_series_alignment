import ast
import json
import sqlite3
from flask import request
from grasp.models import VerifData, CounterData, HistoricData
from grasp import db

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tslearn.metrics import dtw_path
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler




#############################################
##        FONCTIONS DE L APPLICATION       ##
#############################################



# TEXT_TO_NAN
def text_to_nan(cell):
    """Goal: Convert a cell value to a float, or return 'pd.NA' if conversion fails."""

    try:
        # Attempt to convert the cell value to a float.
        return float(cell)
    except:
        # If conversion fails, return 'pd.NA'.
        return pd.NA


# TEXT_TO_FLOAT32
def text_to_float32(value):
    try:
        return np.float32(value)
    except (ValueError, TypeError):
        return np.nan


def convert_columns(df):
    print('DO I REDUCE DIMENSIONS?')
    for col in df.columns:
        max_values = df[col].abs().max()
        if max_values > 65500:
            df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype(np.float16)
    
    return df


# READ_COUNTER
def read_counter():
    """Requête CounterData pour connaître à quel étape de la base de données il faut travailler"""
    counter_entry = CounterData.query.first()
    return counter_entry.counter if counter_entry else 0


# DECREASE_COUNTER
def decrease_counter():
    """Ajoute -1 à COunterData"""
    counter_entry = CounterData.query.first()
    if counter_entry :
        counter_entry.counter -= 1
        db.session.commit()


# INCREMENT_COUNTER
def increment_counter(init_db):
    """Increment the CounterData by +1 (or create it with a value of 1 if it's empty)."""
    counter_entry = CounterData.query.first()
    if counter_entry:
        counter_entry.counter += 1
    else:
        count = count_tables(init_db)

        new_counter_entry = CounterData(counter=1, max_counter=count)
        db.session.add(new_counter_entry)
    db.session.commit()



# COUNT_TABLES
def count_tables(db_path):
    """Compte le nombre de df dans la base de données"""
    
    # Connect to the SQLite database specified by 'db_path'
    conn = sqlite3.connect(db_path)
    
    # Create a cursor to interact with the database
    cursor = conn.cursor()
    
    # Execute an SQL query to count the number of tables in the database
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
    
    # Fetch the result of the query, which is a tuple, and get the first element
    # This element represents the count of tables in the database
    table_count = cursor.fetchone()[0]
    
    # Close the database connection to release resources
    conn.close()
    
    # Return the count of tables in the database
    return table_count



# REORDER_TABLES
def reorder_tables(reference_db, target_db):
    """Réarrange l'ordre des df dans la base de données dans la target en fonction de la référence"""
    
    # Connect to the reference and target databases
    reference_conn = sqlite3.connect(reference_db)
    target_conn = sqlite3.connect(target_db)
    
    # Create a cursor for the reference database and fetch table names
    reference_cursor = reference_conn.cursor()
    reference_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    reference_table_names = [table[0] for table in reference_cursor.fetchall()]

    # Within a transaction in the target database:
    with target_conn:
        # 1. Create temporary tables in the target database
        # 2. Drop original tables in the target database
        for table_name in reference_table_names:
            target_conn.execute(f'CREATE TABLE _temp_{table_name} AS SELECT * FROM {table_name}')
            target_conn.execute(f'DROP TABLE {table_name}')
        
        # 3. Rename temporary tables back to their original names
        for table_name in reference_table_names:
            target_conn.execute(f'ALTER TABLE _temp_{table_name} RENAME TO {table_name}')

    # Close the database connections
    reference_conn.close()
    target_conn.close()



# OVERALL_PLOT
def overall_plot(database, dropdown_value=None):
    """Réalise l'affichage graphique de tous les df modifiés par l'utilisateur"""

    # Connect to the SQLite database specified by 'database'
    conn = sqlite3.connect(database)

    # Fetch the names of tables in the database
    table_names = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [name[0] for name in table_names]

    # Initialize variables for plot data and dropdown options
    fig_json = None
    dropdown_options = None
    selected_variable = None

    if table_names:
        # Find shared columns among all tables
        shared_columns = set()
        for table_name in table_names:
            query = f"PRAGMA table_info({table_name})"
            columns = conn.execute(query).fetchall()
            if not shared_columns:
                shared_columns = set(col[1] for col in columns[2:])
            else:
                shared_columns &= set(col[1] for col in columns[2:])

        # Create dropdown options based on shared columns
        dropdown_options = [{'label': column, 'value': column} for column in shared_columns]

        # Set the selected variable based on dropdown_value or default
        if dropdown_value:
            selected_variable = dropdown_value
        else:
            # If dropdown_value is not provided, attempt to get it from the request or use the first shared column
            selected_variable = request.form.get('variable-dropdown1') or list(shared_columns)[0]

        # Create a subplot figure for multiple plots
        fig = make_subplots(rows=len(table_names), cols=1, shared_xaxes=True, vertical_spacing=0)

        # Create traces for each table and add them to the subplot
        for i, table_name in enumerate(table_names):
            query = f"SELECT {columns[1][1]}, {selected_variable} FROM {table_name}"
            data = conn.execute(query).fetchall()

            x_vals = [row[0] for row in data]
            y_vals = [row[1] for row in data]

            color = px.colors.qualitative.Safe[i % len(px.colors.qualitative.Safe)]

            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color=color), name=table_name), row=i+1, col=1)

        # Update layout and appearance of the plot
        fig.update_layout(
            template="simple_white",
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(showline=True, zeroline=False),
            yaxis=dict(showline=True, zeroline=False),
            margin=dict(l=0, r=0, t=0.2, b=0),
        )

        # Update axis visibility for a clean appearance
        for i in range(1, len(table_names) + 1):
            fig.update_yaxes(showline=False, showticklabels=False, row=i, col=1, color="white")
            fig.update_xaxes(showline=False, showticklabels=False, row=i, col=1, color="white")

        fig.update_xaxes(showline=True, showticklabels=True, row=len(table_names), col=1, color="black", nticks=10)
        fig.update_yaxes(showline=False, showticklabels=False, row=len(table_names), col=1)

        # Convert the plot to JSON format for rendering
        fig_json = json.dumps(fig.to_dict())

        return fig_json, dropdown_options



# FOCUS_PLOT
def focus_plot(init_database, step, global_constraint=None, sakoe_chiba_radius=None, 
               itakura_max_slope=None, align=None, dropdown_value=None, selected_vars=None):
    """Réalise l'affichage graphique du focus sur les 2 df à manipuler par l'utilisateur"""

    # Connection à la base de données initiale
    conn1 = sqlite3.connect(init_database)
    cursor = conn1.cursor()
    # Récupération des 2 tables à partir de l'étape 'step' (qui défini l'avancement dans les séquences)
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' LIMIT 2 OFFSET {step}")
    # On récupère les noms des tables de la bd
    table_names = [row[0] for row in cursor.fetchall()]
    
    # selected_vars sont les variables à utiliser pour aligner, on l'assigne à col_names si elle existe
    col_names = None
    if selected_vars :
        col_names = ast.literal_eval(selected_vars)

    # Liste des tables transformées en df
    dfs = []
    for table_name in table_names:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn1)
        dfs.append(df)
    df1 = dfs[0]
    df2 = dfs[1]

    # Connection à la BD temporary_data, remplaceement des tables
    # temporary_data sera lue plus tard par modified_plot
    conn3 = sqlite3.connect('temporary_data.db')
    df1.to_sql(table_names[0], conn3, if_exists='replace', index=False)
    df2.to_sql(table_names[1], conn3, if_exists='replace', index=False)
    conn3.close()

    # Creation de variables
    fig_json = None
    dropdown_options = None
    var1_dtw = None
    var2_dtw = None
    # dropdown_value = la variable à afficher
    if dropdown_value :
        selected_variable = dropdown_value
    else :
        selected_variable = None

    # Recherche des variables communes entre les 2 tables
    if table_names :
        shared_columns = set()
        for table_name in table_names:
            # Query dans la bd initiale
            query = f"PRAGMA table_info({table_name})"
            columns = conn1.execute(query).fetchall()
            if not shared_columns:
                shared_columns = set(col[1] for col in columns[2:])
            else:
                shared_columns &= set(col[1] for col in columns[2:])
        # Mise en forme
        dropdown_options = [{'label': column, 'value': column} for column in shared_columns]

        # Si l'utilisateur a décider d'aligner les données
        if align :
            # Si pas d'hyperparamètre noté pas l'utilisateur -> None
            # Sinon on change en str()
            if global_constraint == '' : global_constraint = None 
            else : global_constraint = str(global_constraint)
            if sakoe_chiba_radius == '' : sakoe_chiba_radius = None 
            else : sakoe_chiba_radius = int(sakoe_chiba_radius)
            if itakura_max_slope == '' : itakura_max_slope = None
            else : itakura_max_slope = float(itakura_max_slope)
            
            # Si VerifData n'est pas vide
            db_record = VerifData.query.first()
            if db_record :
                # SI JE CHANGE DE VARIABLE SANS CHANGER DE PARAMETRE

                # Si les hyperparam de l'utilisateur sont les même que dans la base de mémoire (VérifData)
                if (global_constraint == db_record.global_constraint) and \
                    (sakoe_chiba_radius == db_record.sakoe_chiba_radius) and \
                    (itakura_max_slope == db_record.itakura_max_slope) and \
                    (selected_vars == db_record.selected_vars):
                    print('CHGT VAR / SAME PARAMS')
                    
                    # Load les fonctions de transformation dtw des séquences précédentes
                    var1_dtw = json.loads(db_record.var1_dtw)
                    var2_dtw = json.loads(db_record.var2_dtw)

                # SI JE CHANGE DE PARAMETRE
                else :  
                    print('CHGT VAR / CHGT PARAMS')
                    # Si l'utilisateur a choisi des variables sur lesquels aligner on sélectionne ces colonnes
                    if col_names :
                        df1_inter = df1.loc[:, ~df1.columns.isin(col_names)]
                        df2_inter = df2.loc[:, ~df2.columns.isin(col_names)]
                    # Sinon on garde toutes les variables
                    else :
                        df1_inter, df2_inter = df1, df2
                    # DTW sur les données sélectionnées
                    var1_dtw, var2_dtw = multivariate_alignment_v2(df1_inter, df2_inter, global_constraint=global_constraint, 
                                                                   sakoe_chiba_radius=sakoe_chiba_radius, 
                                                                   itakura_max_slope=itakura_max_slope)                  

                    # On enregistre les hyperparam dans la db de mémoire (pour les futurs call de la fonction)
                    db_record.global_constraint = global_constraint
                    db_record.sakoe_chiba_radius = sakoe_chiba_radius
                    db_record.itakura_max_slope = itakura_max_slope
                    db_record.selected_vars = selected_vars
                    db_record.var1_dtw = json.dumps(var1_dtw)
                    db_record.var2_dtw = json.dumps(var2_dtw)
                    db.session.commit()

                    # Si l'user a sélectionné une variable à afficher, sinon on prend la première variable de la table
                    if dropdown_value :
                        selected_variable = dropdown_value
                    else :
                        selected_variable = list(shared_columns)[0]                    


            # SI C'EST LA PREMIERE FOIS QUE L'ON ALIGNE
            else :   
                print('FIRST TIME')
                if col_names :
                    df1_inter = df1.loc[:, ~df1.columns.isin(col_names)]
                    df2_inter = df2.loc[:, ~df2.columns.isin(col_names)]
                else :
                    df1_inter, df2_inter = df1, df2                
                var1_dtw, var2_dtw = multivariate_alignment_v2(df1_inter, df2_inter, global_constraint=global_constraint, 
                                                               sakoe_chiba_radius=sakoe_chiba_radius, 
                                                               itakura_max_slope=itakura_max_slope)             

                # Premier enregistrement des paramètres dans la base mémoire
                new_record = VerifData(
                    global_constraint=global_constraint,
                    sakoe_chiba_radius=sakoe_chiba_radius,
                    itakura_max_slope=itakura_max_slope,
                    selected_vars=str(col_names),
                    var1_dtw=json.dumps(var1_dtw),
                    var2_dtw=json.dumps(var2_dtw))
                db.session.add(new_record)             
                db.session.commit()

                selected_variable = dropdown_value

            # Transformation des données pour l'affichage graphique
            df1 = df1.iloc[var1_dtw].reset_index(drop=True)
            df2 = df2.iloc[var2_dtw].reset_index(drop=True)
            df2.iloc[:, 1] = df1.iloc[:, 1]

            # Mise à jour de temporary_data
            conn3 = sqlite3.connect('temporary_data.db')
            cursor_target = conn3.cursor()
            cursor_target.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;")
            table_name_row = cursor_target.fetchone()
            if table_name_row:
                table_name = table_name_row[0]
                cursor_target.execute(f"DROP TABLE IF EXISTS {table_name};")
            df1.to_sql(table_names[0], conn3, if_exists='replace', index=False)
            df2.to_sql(table_names[1], conn3, if_exists='replace', index=False)
            
            conn3.close()
            conn1.close()

        # Si pas de variable à afficher par l'utilisateur alors on regarde le boutton variable-dropdown2
        if not selected_variable :
            if request.form.get('variable-dropdown2') :
                selected_variable = request.form.get('variable-dropdown2')
            else : 
                selected_variable = list(shared_columns)[0]    

        # list des axes x et y
        y_vals1 = df1[selected_variable].tolist()
        x_vals1 = df1.iloc[:, 1].tolist()
        y_vals2 = df2[selected_variable].tolist()
        x_vals2 = df2.iloc[:, 1].tolist()        

        # Figure 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals1, y=y_vals1, mode='lines', name=table_names[0], 
                                 line=dict(color=px.colors.qualitative.Safe[step % len(px.colors.qualitative.Safe)])))
        fig.add_trace(go.Scatter(x=x_vals2, y=y_vals2, mode='lines', name=table_names[1], 
                                 line=dict(color=px.colors.qualitative.Safe[step+1 % len(px.colors.qualitative.Safe)])))
        fig.update_layout(
            template="simple_white",
            paper_bgcolor='white',  
            plot_bgcolor='white',
            xaxis=dict(showline=True, zeroline=False),
            yaxis=dict(showline=True, zeroline=False), 
            margin=dict(l=0, r=0, t=0, b=0),
        )
        fig.update_xaxes(showline=True, showticklabels=True, color="black", nticks=10)
        fig.update_yaxes(showline=True, showticklabels=True, color="black", nticks=7)
        fig_json = json.dumps(fig.to_dict())


        #Remplir HistoricData pour avoir en mémoire la taille de df2 et les dtw_path
        history = HistoricData.query.all()
        if not history :
            new_row = HistoricData()
            if not var2_dtw :
                print('DTW not added')
                new_row.var2_dtw = json.dumps(list(range(len(df2))))
                new_row.lgth_last_elem=len(df2)
                new_row.min_depth=np.min(df2.iloc[:,1])
                new_row.max_depth=np.max(df2.iloc[:,1])
            else :
                print('DTW added')
                new_row.var2_dtw = json.dumps(var2_dtw)
                new_row.lgth_last_elem=len(df2)
                new_row.min_depth=np.min(df2.iloc[:,1])
                new_row.max_depth=np.max(df2.iloc[:,1])
            db.session.add(new_row)
        else :
            if len(history) > step :
                step_row = history[step]
                if not var2_dtw:
                    print('DTW not added 2')
                    step_row.var2_dtw = json.dumps(list(range(len(df2))))
                    step_row.lgth_last_elem=len(df2)
                    step_row.min_depth=np.min(df2.iloc[:,1])
                    step_row.max_depth=np.max(df2.iloc[:,1])
                else:
                    print('DTW added 2')
                    step_row.var2_dtw = json.dumps(var2_dtw)
                    step_row.min_depth=np.min(df2.iloc[:,1])
                    step_row.max_depth=np.max(df2.iloc[:,1])       
            else :
                print('STEP', step, len(history))
                if not var2_dtw:
                    print('DTW not added 3')
                    new_row = HistoricData(lgth_last_elem=len(df2), var2_dtw=json.dumps(list(range(len(df2)))),
                                           min_depth=np.min(df2.iloc[:,1]), max_depth=np.max(df2.iloc[:,1]))
                else :
                    print('DTW added 3')
                    new_row = HistoricData(lgth_last_elem=len(df2), var2_dtw=json.dumps(var2_dtw),
                                           min_depth=np.min(df2.iloc[:,1]), max_depth=np.max(df2.iloc[:,1]))
                db.session.add(new_row)
        db.session.commit()  

        return fig_json, dropdown_options


# FOCUS_PLOT
def focus_plot2(init_database, step, global_constraint=None, sakoe_chiba_radius=None, 
               itakura_max_slope=None, align=None, dropdown_value=None, selected_vars=None):
    """
    Réalise l'affichage graphique du focus sur les 2 df à manipuler par l'utilisateur.
    Focus_plot2 tourne lorsque l'utilisateur choisi l'Option 2 (càd qu'il aligne entre 
    la séquence de référence (0) et les autres séquences une à une)
    """

    # Connection à la base de données initiale
    conn1 = sqlite3.connect(init_database)
    cursor = conn1.cursor()
    table_names = []
    # Récupération de la table 0 et step+1
    for i in [0, step+1]:
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' LIMIT 1 OFFSET {i}")
        table_names.append(cursor.fetchall()[0][0])

    # selected_vars sont les variables à utiliser pour aligner, on l'assigne à col_names si elle existe
    col_names = None
    if selected_vars :
        col_names = ast.literal_eval(selected_vars)

    # Liste des tables transformées en df
    dfs = []
    for table_name in table_names:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn1)
        dfs.append(df)
    df1 = dfs[0]
    df2 = dfs[1]

    # Connection à la BD temporary_data, remplaceement des tables
    # temporary_data sera lue plus tard par modified_plot
    conn3 = sqlite3.connect('temporary_data.db')
    df1.to_sql(table_names[0], conn3, if_exists='replace', index=False)
    df2.to_sql(table_names[1], conn3, if_exists='replace', index=False)
    conn3.close()

    # Creation de variables
    fig_json = None
    dropdown_options = None
    var2_dtw = None
    # dropdown_value = la variable à afficher
    if dropdown_value :
        selected_variable = dropdown_value
    else :
        selected_variable = None

    # Recherche des variables communes entre les 2 tables
    if table_names :
        shared_columns = set()
        for table_name in table_names:
            # Query dans la bd initiale
            query = f"PRAGMA table_info({table_name})"
            columns = conn1.execute(query).fetchall()
            if not shared_columns:
                shared_columns = set(col[1] for col in columns[2:])
            else:
                shared_columns &= set(col[1] for col in columns[2:])
        # Mise en forme
        dropdown_options = [{'label': column, 'value': column} for column in shared_columns]

        # Si l'utilisateur a décidé d'aligner les données
        if align :
            # Si pas d'hyperparamètre noté pas l'utilisateur -> None
            # Sinon on change en str()
            if global_constraint == '' : global_constraint = None 
            else : global_constraint = str(global_constraint)
            if sakoe_chiba_radius == '' : sakoe_chiba_radius = None 
            else : sakoe_chiba_radius = int(sakoe_chiba_radius)
            if itakura_max_slope == '' : itakura_max_slope = None
            else : itakura_max_slope = float(itakura_max_slope)

            # Si VerifData n'est pas vide
            db_record = VerifData.query.first()
            if db_record :
                # SI JE CHANGE DE VARIABLE SANS CHANGER DE PARAMETRE

                # Si les hyperparam de l'utilisateur sont les même que dans la base de mémoire (VérifData)
                if (global_constraint == db_record.global_constraint) and \
                    (sakoe_chiba_radius == db_record.sakoe_chiba_radius) and \
                    (itakura_max_slope == db_record.itakura_max_slope) and \
                    (selected_vars == db_record.selected_vars):
                    print('CHGT VAR / SAME PARAMS')
                    
                    # Load les fonctions de transformation dtw des séquences précédentes
                    var1_dtw = json.loads(db_record.var1_dtw)
                    var2_dtw = json.loads(db_record.var2_dtw)

                # SI JE CHANGE DE PARAM
                else :  
                    print('CHGT VAR / CHGT PARAMS')
                    # Si l'utilisateur a choisi des variables sur lesquels aligner on sélectionne ces colonnes
                    if col_names :
                        df1_inter = df1.loc[:, ~df1.columns.isin(col_names)]
                        df2_inter = df2.loc[:, ~df2.columns.isin(col_names)]
                    # Sinon on garde toutes les variables
                    else :
                        df1_inter, df2_inter = df1, df2
                    # DTW sur les données sélectionnées
                    var1_dtw, var2_dtw = multivariate_alignment_v2(df1_inter, df2_inter, global_constraint=global_constraint, 
                                                                   sakoe_chiba_radius=sakoe_chiba_radius, 
                                                                   itakura_max_slope=itakura_max_slope)                  

                    # On enregistre les hyperparam dans la db de mémoire (pour les futurs call de la fonction)
                    db_record.global_constraint = global_constraint
                    db_record.sakoe_chiba_radius = sakoe_chiba_radius
                    db_record.itakura_max_slope = itakura_max_slope
                    db_record.selected_vars = selected_vars
                    db_record.var1_dtw = json.dumps(var1_dtw)
                    db_record.var2_dtw = json.dumps(var2_dtw)
                    db.session.commit()

                    # Si l'user a sélectionné une variable à afficher, sinon on prend la première variable de la table
                    if dropdown_value :
                        selected_variable = dropdown_value
                    else :
                        selected_variable = list(shared_columns)[0]                    


            # SI C'EST LA PREMIERE FOIS QUE L'ON ALIGNE
            else :   
                print('FIRST TIME')
                if col_names :
                    df1_inter = df1.loc[:, ~df1.columns.isin(col_names)]
                    df2_inter = df2.loc[:, ~df2.columns.isin(col_names)]
                else :
                    df1_inter, df2_inter = df1, df2                
                var1_dtw, var2_dtw = multivariate_alignment_v2(df1_inter, df2_inter, global_constraint=global_constraint, 
                                                               sakoe_chiba_radius=sakoe_chiba_radius, 
                                                               itakura_max_slope=itakura_max_slope)             

                # Premier enregistrement des paramètres dans la base mémoire
                new_record = VerifData(
                    global_constraint=global_constraint,
                    sakoe_chiba_radius=sakoe_chiba_radius,
                    itakura_max_slope=itakura_max_slope,
                    selected_vars=str(col_names),
                    var1_dtw=json.dumps(var1_dtw),
                    var2_dtw=json.dumps(var2_dtw))
                db.session.add(new_record)             
                db.session.commit()

                selected_variable = dropdown_value

            # Transformation des données pour l'affichage graphique
            df1 = df1.iloc[var1_dtw].reset_index(drop=True)
            df2 = df2.iloc[var2_dtw].reset_index(drop=True)
            df2.iloc[:, 1] = df1.iloc[:, 1]

            # Mise à jour de temporary_data
            conn3 = sqlite3.connect('temporary_data.db')
            cursor_target = conn3.cursor()
            cursor_target.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;")
            table_name_row = cursor_target.fetchone()
            if table_name_row:
                table_name = table_name_row[0]
                cursor_target.execute(f"DROP TABLE IF EXISTS {table_name};")
            df1.to_sql(table_names[0], conn3, if_exists='replace', index=False)
            df2.to_sql(table_names[1], conn3, if_exists='replace', index=False)
            
            conn3.close()
            conn1.close()

        # Si pas de variable à afficher par l'utilisateur alors on regarde le boutton variable-dropdown2
        if not selected_variable :
            if request.form.get('variable-dropdown2') :
                selected_variable = request.form.get('variable-dropdown2')
            else : 
                selected_variable = list(shared_columns)[0]    

        # list des axes x et y
        y_vals1 = df1[selected_variable].tolist()
        x_vals1 = df1.iloc[:, 1].tolist()
        y_vals2 = df2[selected_variable].tolist()
        x_vals2 = df2.iloc[:, 1].tolist()        

        # Figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals1, y=y_vals1, mode='lines', name=table_names[0], 
                                 line=dict(color=px.colors.qualitative.Safe[step % len(px.colors.qualitative.Safe)])))
        fig.add_trace(go.Scatter(x=x_vals2, y=y_vals2, mode='lines', name=table_names[1], 
                                 line=dict(color=px.colors.qualitative.Safe[step+1 % len(px.colors.qualitative.Safe)])))
        fig.update_layout(
            template="simple_white",
            paper_bgcolor='white',
            plot_bgcolor='white', 
            xaxis=dict(showline=True, zeroline=False),
            yaxis=dict(showline=True, zeroline=False),
            margin=dict(l=0, r=0, t=0, b=0),  # Adjust the margin as needed
        )
        fig.update_xaxes(showline=True, showticklabels=True, color="black", nticks=10)
        fig.update_yaxes(showline=True, showticklabels=True, color="black", nticks=7)
        fig_json = json.dumps(fig.to_dict())

        return fig_json, dropdown_options



############################################
##        FONCTIONS DE L ALGORITHME       ##
############################################


#INTERPOLATION
def interpolation(lgth, df, lst_elem, min_depth, max_depth):
    # Define the interpolation function that resizes a DataFrame to a desired length

    max_elem = np.max(lst_elem)
    min_elem = np.min(lst_elem)

    desired_length = lgth  # Desired length for the interpolated DataFrame
    # Create interpolation functions for each column in the input DataFrame using linear interpolation
    interp_funcs = {
        col: interp1d(np.arange(len(df)), df[col], kind='linear') for col in df.columns
    }
    # Generate new indices for the interpolated data based on the desired length
    new_indices = np.linspace(0, len(df) - 1, desired_length)
    # Interpolate data for each column using the interpolation functions
    interpolated_data = {col: interp_func(new_indices) for col, interp_func in interp_funcs.items()}
    # Create a new DataFrame with the interpolated data
    interpolated_df = pd.DataFrame(interpolated_data)

    scaler = MinMaxScaler(feature_range=(min_elem, max_elem))
    interpolated_df.iloc[:,0] = scaler.fit_transform(np.array(interpolated_df.iloc[:, 0]).reshape(-1, 1))
    interpolated_df.iloc[:, 0] = np.round(interpolated_df.iloc[:, 0]).astype(int)
    scaler2 = MinMaxScaler(feature_range=(float(min_depth), float(max_depth)))
    interpolated_df.iloc[:,1] = scaler2.fit_transform(np.array(interpolated_df.iloc[:, 1]).reshape(-1, 1))

    return interpolated_df  # Return the resized DataFrame after linear interpolation



# MULTIVARIATE_ALIGNMENT
def multivariate_alignment(var_dtw):
    # Define the multivariate alignment function that computes a transformed dataset

    # Extract weights and paths from the input dictionary
    weights = [var_dtw[i]['ratio'] for i in var_dtw.keys()]
    paths = [var_dtw[i]['path'] for i in var_dtw.keys()]

    # Find unique values in the 'init' column of the paths DataFrames
    vals = []
    for i in range(len(paths)):
        a = pd.DataFrame(paths[i], columns=['init', 'transf'])
        vals.extend(a['init'].unique())
    vals = np.unique(vals)

    # Compute transformed data based on unique 'init' values
    transform_data = []
    for j in vals:
        val = 0
        for i in range(len(paths)):
            df = pd.DataFrame(paths[i], columns=['init', 'transf'])
            val += df[df['init'] == j]['transf'].mean() * weights[i]
        val = round(val / sum(weights))
        transform_data.append(val)

    return transform_data  # Return the computed transformed dataset



# MULTIVARIATE_ALIGNMENT_V2
def multivariate_alignment_v2(df1, df2, global_constraint='itakura', sakoe_chiba_radius=None, 
                              itakura_max_slope=None):
    # Define the multivariate alignment function version 2 that computes alignment paths
    
    df1 = convert_columns(df1)
    df2 = convert_columns(df2)
    print(df2.dtypes)
    # Compute the alignment path between two input DataFrames
    path = dtw_path(df1.values, df2.values, global_constraint=global_constraint, 
                    sakoe_chiba_radius=sakoe_chiba_radius, itakura_max_slope=itakura_max_slope)
    
    # Extract the first elements and second elements from the alignment path
    premiers_elements = [t[0] for t in list(path)[0]]
    seconds_elements = [t[1] for t in list(path)[0]]

    return premiers_elements, seconds_elements  # Return the alignment path elements
