import ast
import json
import sqlite3
from flask import request
from grasp.models import VerifData, CounterData, HistoricData
from grasp import db

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dtaidistance import dtw
from tslearn.metrics import dtw_path
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d



#############################################
##        FONCTIONS DE L APPLICATION       ##
#############################################

#ALIGN_FROM_HISTORY
def align_from_history(db, step):
    """Accumule les fonctions d'alignement dans une base de données pour pouvoir les appliquer lors de next_data"""
    conn1 = sqlite3.connect(db)
    query = "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1"
    table_info = conn1.execute(query).fetchone()
    if table_info:
        table_name = table_info[0]
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn1)
        conn1.close()


        history = HistoricData.query.limit(step).all()

        for row in history :
            var2_dtw = ast.literal_eval(row.var2_dtw)
            df = df.iloc[var2_dtw].reset_index(drop=True)

        conn2 = sqlite3.connect('transformed_data.db')
        df.to_sql('transformed_table', conn2, if_exists='replace', index=False)
        conn2.close()

        


#TEXT_TO_NAN
def text_to_nan(cell):
    try:
        return float(cell)
    except:
        return pd.NA


#READ_COUNTER
def read_counter():
    """Requête CounterData pour connaître à quel étape de la base de données il faut travailler"""
    counter_entry = CounterData.query.first()
    return counter_entry.counter if counter_entry else 0


#DECREASE_COUNTER
def decrease_counter():
    """Ajoute -1 à COunterData"""
    counter_entry = CounterData.query.first()
    if counter_entry :
        counter_entry.counter -= 1
        db.session.commit()


#INCREMENT_COUNTER
def increment_counter(init_db):
    """Ajoute +1 à CounterData (ou crée 1 si CounterData est vide)"""
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
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
    table_count = cursor.fetchone()[0]
    conn.close()
    return table_count


# COPY_TABLES
def copy_tables(source_db, target_db):
    """Copie les df dont les nom sont communs entre 2 bases de données : la source et la target"""
    source_conn = sqlite3.connect(source_db)
    target_conn = sqlite3.connect(target_db)

    source_cursor = source_conn.cursor()
    source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    source_table_names = {table[0] for table in source_cursor.fetchall()}

    target_cursor = target_conn.cursor()
    target_cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    target_tables = target_cursor.fetchall()
    target_table_names = {table[0] for table in target_tables}
    target_table_schemas = {table[0]: table[1] for table in target_tables}

    shared_table_names = source_table_names.intersection(target_table_names)

    for table_name in shared_table_names:
        if table_name != 'sqlite_sequence':
            if table_name in target_table_schemas:
                target_cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
            source_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            source_table_schema = source_cursor.fetchone()[0]
            target_cursor.execute(source_table_schema)
            source_cursor.execute(f'SELECT * FROM {table_name}')
            data_to_copy = source_cursor.fetchall()
            target_cursor.executemany(f'INSERT INTO {table_name} VALUES ({", ".join(["?"] * len(data_to_copy[0]))})', data_to_copy)

    target_conn.commit()
    target_conn.close()
    source_conn.close()


#REORDER_TABLES
def reorder_tables(reference_db, target_db):
    """Réarrange l'ordre des df dans la base de données dans la target en fonction de la référence"""
    reference_conn = sqlite3.connect(reference_db)
    target_conn = sqlite3.connect(target_db)
    reference_cursor = reference_conn.cursor()
    reference_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    reference_table_names = [table[0] for table in reference_cursor.fetchall()]

    with target_conn:
        for table_name in reference_table_names:
            target_conn.execute(f'CREATE TABLE _temp_{table_name} AS SELECT * FROM {table_name}')
            target_conn.execute(f'DROP TABLE {table_name}')
        for table_name in reference_table_names:
            target_conn.execute(f'ALTER TABLE _temp_{table_name} RENAME TO {table_name}')

    reference_conn.close()
    target_conn.close()


#READ_TABLES
def read_tables(database):
    """INUTILE"""
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name LIMIT 2")
    table_names = [row[0] for row in cursor.fetchall()]

    data_objects = {}
    
    for table_name in table_names:
        cursor.execute(f'SELECT * FROM {table_name}')
        data = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        columns = [col for col in columns if col.lower() != 'index']
        data_objects[table_name] = [dict(zip(columns, row)) for row in data]

    connection.close()

    return data_objects, table_names


# OVERALL_PLOT
def overall_plot(database, dropdown_value=None):
    """Réalise l'affichage graphique de tous les df modifiés par l'utilisateur"""
    conn = sqlite3.connect(database)

    table_names = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [name[0] for name in table_names]

    fig_json = None
    dropdown_options = None
    selected_variable = None

    if table_names :
        shared_columns = set()
        for table_name in table_names:
            query = f"PRAGMA table_info({table_name})"
            data = conn.execute(query).fetchall()
            columns = conn.execute(query).fetchall()
            if not shared_columns:
                shared_columns = set(col[1] for col in columns[2:])
            else:
                shared_columns &= set(col[1] for col in columns[2:])

        dropdown_options = [{'label': column, 'value': column} for column in shared_columns]

        if dropdown_value :
            selected_variable = dropdown_value
        else :
            selected_variable = request.form.get('variable-dropdown1') or list(shared_columns)[0]
        
        fig = make_subplots(rows=len(table_names), cols=1, shared_xaxes=True, vertical_spacing=0)

        for i, table_name in enumerate(table_names):
            query = f"SELECT {columns[1][1]}, {selected_variable} FROM {table_name}"
            data = conn.execute(query).fetchall()

            x_vals = [row[0] for row in data]
            y_vals = [row[1] for row in data]

            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=table_name), row=i+1, col=1)

        fig.update_layout(
            template="simple_white",
            paper_bgcolor='white', 
            plot_bgcolor='white',  
            xaxis=dict(showline=True, zeroline=False),
            yaxis=dict(showline=True, zeroline=False),
            margin=dict(l=0, r=0, t=0.2, b=0),  
        )
        for i in range(1, len(table_names)+1):
            fig.update_yaxes(showline=False, showticklabels=False, row=i, col=1, color="white")
            fig.update_xaxes(showline=False, showticklabels=False, row=i, col=1, color="white")
        fig.update_xaxes(showline=True, showticklabels=True, row=len(table_names), col=1, color="black", dtick=10)
        fig.update_yaxes(showline=False, showticklabels=False, row=len(table_names), col=1)
        fig_json = json.dumps(fig.to_dict())

        return fig_json, dropdown_options


# FOCUS_PLOT
def focus_plot(init_database, step, global_constraint=None, sakoe_chiba_radius=None, 
               itakura_max_slope=None, align=None, dropdown_value=None, selected_vars=None):
    """Réalise l'affichage graphique du focus sur les 2 df à manipuler par l'utilisateur"""

    # Connection à la base de données initiale
    conn1 = sqlite3.connect(init_database)
    cursor = conn1.cursor()
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' LIMIT 2 OFFSET {step}")
    table_names = [row[0] for row in cursor.fetchall()]
    
    col_names = None
    if selected_vars :
        col_names = ast.literal_eval(selected_vars)

    dfs = []
    for table_name in table_names:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn1)
        dfs.append(df)
    df1 = dfs[0]
    df2 = dfs[1]

    # Connection à la BD à envoyer à modified_plot
    conn3 = sqlite3.connect('temporary_data.db')
    df1.to_sql(table_names[0], conn3, if_exists='replace', index=False)
    df2.to_sql(table_names[1], conn3, if_exists='replace', index=False)
    conn3.close()

    # Creation de variables
    fig_json = None
    dropdown_options = None
    var2_dtw = None
    if dropdown_value :
        selected_variable = dropdown_value
    else :
        selected_variable = None

    # Recherche des variables communes
    if table_names :
        shared_columns = set()
        for table_name in table_names:
            query = f"PRAGMA table_info({table_name})"
            columns = conn1.execute(query).fetchall()
            if not shared_columns:
                shared_columns = set(col[1] for col in columns[2:])
            else:
                shared_columns &= set(col[1] for col in columns[2:])

        dropdown_options = [{'label': column, 'value': column} for column in shared_columns]

        if align :
            db_record = VerifData.query.first()

            if global_constraint == '' : global_constraint = None 
            else : global_constraint = str(global_constraint)
            if sakoe_chiba_radius == '' : sakoe_chiba_radius = None 
            else : sakoe_chiba_radius = int(sakoe_chiba_radius)
            if itakura_max_slope == '' : itakura_max_slope = None
            else : itakura_max_slope = float(itakura_max_slope)
            
            if db_record :
                # SI JE CHANGE DE VARIABLE SANS CHANGER DE PARAM
                if (global_constraint == db_record.global_constraint) and \
                    (sakoe_chiba_radius == db_record.sakoe_chiba_radius) and \
                    (itakura_max_slope == db_record.itakura_max_slope) and \
                    (selected_vars == db_record.selected_vars):
                    print('CHGT VAR / SAME PARAMS')
                    
                    var1_dtw = json.loads(db_record.var1_dtw)
                    var2_dtw = json.loads(db_record.var2_dtw)

                # SI JE CHANGE DE PARAM
                else :  
                    print('CHGT VAR / CHGT PARAMS')
                    if col_names :
                        df1_inter = df1.loc[:, ~df1.columns.isin(col_names)]
                        df2_inter = df2.loc[:, ~df2.columns.isin(col_names)]
                    else :
                        df1_inter, df2_inter = df1, df2
                    var1_dtw, var2_dtw = multivariate_alignment_v2(df1_inter, df2_inter, global_constraint=global_constraint, 
                                                                   sakoe_chiba_radius=sakoe_chiba_radius, 
                                                                   itakura_max_slope=itakura_max_slope)                  

                    db_record.global_constraint = global_constraint
                    db_record.sakoe_chiba_radius = sakoe_chiba_radius
                    db_record.itakura_max_slope = itakura_max_slope
                    db_record.selected_vars = selected_vars
                    db_record.var1_dtw = json.dumps(var1_dtw)
                    db_record.var2_dtw = json.dumps(var2_dtw)
                    db.session.commit()

                    if dropdown_value :
                        selected_variable = dropdown_value
                    else :
                        selected_variable = list(shared_columns)[0]                    


            # SI C'EST LA PREMIERE FOIS
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

            df1 = df1.iloc[var1_dtw].reset_index(drop=True)
            df2 = df2.iloc[var2_dtw].reset_index(drop=True)
            df2.iloc[:, 1] = df1.iloc[:, 1]

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

        if not selected_variable :
            if request.form.get('variable-dropdown2') :
                selected_variable = request.form.get('variable-dropdown2')
            else : 
                selected_variable = list(shared_columns)[0]    

        y_vals1 = df1[selected_variable].tolist()
        x_vals1 = df1.iloc[:, 1].tolist()
        y_vals2 = df2[selected_variable].tolist()
        x_vals2 = df2.iloc[:, 1].tolist()        

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals1, y=y_vals1, mode='lines', name=table_names[0]))
        fig.add_trace(go.Scatter(x=x_vals2, y=y_vals2, mode='lines', name=table_names[1]))
        fig.update_layout(
            paper_bgcolor='white',  
            plot_bgcolor='white', 
            xaxis=dict(showline=True, zeroline=False),
            yaxis=dict(showline=True, zeroline=False),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_xaxes(showline=True)
        fig_json = json.dumps(fig.to_dict())


        #Remplir HistoricData pour avoir en mémoire la taille de df2 et les dtw_path
        history = HistoricData.query.limit(step).all()
        if not history :
            new_row = HistoricData(lgth_last_elem=len(df2))
            if not var2_dtw :
                new_row.var2_dtw = json.dumps(list(range(len(df2))))
            else :
                new_row.var2_dtw = json.dumps(var2_dtw)
            db.session.add(new_row)
        else :
            if len(history) >= step :
                last_row = history[-1]
                last_row.lgth_last_elem = len(df2)
                if not var2_dtw:
                    last_row.var2_dtw = json.dumps(list(range(len(df2))))
                else:
                    last_row.var2_dtw = json.dumps(var2_dtw)       
            else :
                print('STEP', step, len(history))
                if not var2_dtw:
                    new_row = HistoricData(lgth_last_elem=len(df2), var2_dtw=json.dumps(list(range(len(df2)))))
                else :
                    new_row = HistoricData(lgth_last_elem=len(df2), var2_dtw=json.dumps(var2_dtw))
                db.session.add(new_row)
        db.session.commit()   

        return fig_json, dropdown_options


# FOCUS_PLOT
def focus_plot2(init_database, step, global_constraint=None, sakoe_chiba_radius=None, 
               itakura_max_slope=None, align=None, dropdown_value=None, selected_vars=None):
    """Réalise l'affichage graphique du focus sur les 2 df à manipuler par l'utilisateur"""

    # Connection à la base de données initiale
    conn1 = sqlite3.connect(init_database)
    cursor = conn1.cursor()
    table_names = []
    for i in [0, step+1]:
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' LIMIT 1 OFFSET {i}")
        table_names.append(cursor.fetchall()[0][0])
    print(table_names)

    
    col_names = None
    if selected_vars :
        col_names = ast.literal_eval(selected_vars)

    dfs = []
    for table_name in table_names:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn1)
        dfs.append(df)
    df1 = dfs[0]
    df2 = dfs[1]

    # Connection à la BD à envoyer à modified_plot
    conn3 = sqlite3.connect('temporary_data.db')
    df1.to_sql(table_names[0], conn3, if_exists='replace', index=False)
    df2.to_sql(table_names[1], conn3, if_exists='replace', index=False)
    conn3.close()

    # Creation de variables
    fig_json = None
    dropdown_options = None
    var2_dtw = None
    if dropdown_value :
        selected_variable = dropdown_value
    else :
        selected_variable = None

    # Recherche des variables communes
    if table_names :
        shared_columns = set()
        for table_name in table_names:
            query = f"PRAGMA table_info({table_name})"
            columns = conn1.execute(query).fetchall()
            if not shared_columns:
                shared_columns = set(col[1] for col in columns[2:])
            else:
                shared_columns &= set(col[1] for col in columns[2:])

        dropdown_options = [{'label': column, 'value': column} for column in shared_columns]

        if align :
            db_record = VerifData.query.first()

            if global_constraint == '' : global_constraint = None 
            else : global_constraint = str(global_constraint)
            if sakoe_chiba_radius == '' : sakoe_chiba_radius = None 
            else : sakoe_chiba_radius = int(sakoe_chiba_radius)
            if itakura_max_slope == '' : itakura_max_slope = None
            else : itakura_max_slope = float(itakura_max_slope)
            
            if db_record :
                # SI JE CHANGE DE VARIABLE SANS CHANGER DE PARAM
                if (global_constraint == db_record.global_constraint) and \
                    (sakoe_chiba_radius == db_record.sakoe_chiba_radius) and \
                    (itakura_max_slope == db_record.itakura_max_slope) and \
                    (selected_vars == db_record.selected_vars):
                    print('CHGT VAR / SAME PARAMS')
                    
                    var1_dtw = json.loads(db_record.var1_dtw)
                    var2_dtw = json.loads(db_record.var2_dtw)

                # SI JE CHANGE DE PARAM
                else :  
                    print('CHGT VAR / CHGT PARAMS')
                    if col_names :
                        df1_inter = df1.loc[:, ~df1.columns.isin(col_names)]
                        df2_inter = df2.loc[:, ~df2.columns.isin(col_names)]
                    else :
                        df1_inter, df2_inter = df1, df2
                    var1_dtw, var2_dtw = multivariate_alignment_v2(df1_inter, df2_inter, global_constraint=global_constraint, 
                                                                   sakoe_chiba_radius=sakoe_chiba_radius, 
                                                                   itakura_max_slope=itakura_max_slope)                  

                    db_record.global_constraint = global_constraint
                    db_record.sakoe_chiba_radius = sakoe_chiba_radius
                    db_record.itakura_max_slope = itakura_max_slope
                    db_record.selected_vars = selected_vars
                    db_record.var1_dtw = json.dumps(var1_dtw)
                    db_record.var2_dtw = json.dumps(var2_dtw)
                    db.session.commit()

                    if dropdown_value :
                        selected_variable = dropdown_value
                    else :
                        selected_variable = list(shared_columns)[0]                    


            # SI C'EST LA PREMIERE FOIS
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

            df1 = df1.iloc[var1_dtw].reset_index(drop=True)
            df2 = df2.iloc[var2_dtw].reset_index(drop=True)
            df2.iloc[:, 1] = df1.iloc[:, 1]

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

        if not selected_variable :
            if request.form.get('variable-dropdown2') :
                selected_variable = request.form.get('variable-dropdown2')
            else : 
                selected_variable = list(shared_columns)[0]    

        y_vals1 = df1[selected_variable].tolist()
        x_vals1 = df1.iloc[:, 1].tolist()
        y_vals2 = df2[selected_variable].tolist()
        x_vals2 = df2.iloc[:, 1].tolist()        

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals1, y=y_vals1, mode='lines', name=table_names[0]))
        fig.add_trace(go.Scatter(x=x_vals2, y=y_vals2, mode='lines', name=table_names[1]))
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white', 
            xaxis=dict(showline=True, zeroline=False),
            yaxis=dict(showline=True, zeroline=False),
            margin=dict(l=0, r=0, t=0, b=0),  # Adjust the margin as needed
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_xaxes(showline=True)
        fig_json = json.dumps(fig.to_dict())


        #Remplir HistoricData pour avoir en mémoire la taille de df2 et les dtw_path
        history = HistoricData.query.limit(step).all()
        if not history :
            new_row = HistoricData(lgth_last_elem=len(df2))
            if not var2_dtw :
                new_row.var2_dtw = json.dumps(list(range(len(df2))))
            else :
                new_row.var2_dtw = json.dumps(var2_dtw)
            db.session.add(new_row)
        else :
            if len(history) >= step :
                last_row = history[-1]
                last_row.lgth_last_elem = len(df2)
                if not var2_dtw:
                    last_row.var2_dtw = json.dumps(list(range(len(df2))))
                else:
                    last_row.var2_dtw = json.dumps(var2_dtw)       
            else :
                print('STEP', step, len(history))
                if not var2_dtw:
                    new_row = HistoricData(lgth_last_elem=len(df2), var2_dtw=json.dumps(list(range(len(df2)))))
                else :
                    new_row = HistoricData(lgth_last_elem=len(df2), var2_dtw=json.dumps(var2_dtw))
                db.session.add(new_row)
        db.session.commit()   

        return fig_json, dropdown_options



############################################
##        FONCTIONS DE L ALGORITHME       ##
############################################


#INTERPOLATION
def interpolation(lgth, df):
    desired_length = lgth

    interp_funcs = {
        col: interp1d(np.arange(len(df)), df[col], kind='linear') for col in df.columns
    }

    new_indices = np.linspace(0, len(df) - 1, desired_length)

    interpolated_data = {col: interp_func(new_indices) for col, interp_func in interp_funcs.items()}
    interpolated_df = pd.DataFrame(interpolated_data)
    
    return interpolated_df 


# EXTRACT_ELEMENTS
def extract_elements(liste_de_tuples):
    premiers_elements = [t[0] for t in liste_de_tuples]
    seconds_elements = [t[1] for t in liste_de_tuples]
    return premiers_elements, seconds_elements


# UNIVARIATE_ALIGNMENT
def univariate_alignment(df1, df2, global_constraint='itakura', sakoe_chiba_radius=None,
                         itakura_max_slope=None):
    
    var_names = df1.iloc[:, 1:].columns.tolist()

    var_dtw = {}
    scaler = MinMaxScaler()

    for v in var_names:
        data1 = df1[v].values.reshape(-1, 1)
        data2 = df2[v].values.reshape(-1, 1)
        data1 = scaler.fit_transform(data1)
        data2 = scaler.fit_transform(data2)
        
        # DTW pour pour chaque variable
        alignment_before = dtw_path(data1[:, 0], data2[:, 0], global_constraint=global_constraint, 
                                    sakoe_chiba_radius=sakoe_chiba_radius, itakura_max_slope=itakura_max_slope)

        # Index de chaque df
        indices1 = [idx1 for idx1, _ in alignment_before[0]]
        indices2 = [idx2 for _, idx2 in alignment_before[0]]
        
        # Data alignées
        aligned_data1 = data1[indices1]
        aligned_data2 = data2[indices2]
        
        # Calcul de la distance en x entre les 2 data
        x1, x2 = extract_elements(alignment_before[0])
        stretch_cost = sum([abs(a-b) for a ,b in zip(x1,x2)])
        
        # Calcul de la métrique pour évaluer l'importance de la série pour l'alignement multivarié
        ratio = abs((sum(aligned_data1) - sum(aligned_data2))) / (abs((sum(data1) - sum(data2))) *len(indices1)*stretch_cost)
        
        # Conservation des résultats
        var_dtw[v] = {
            'ratio': ratio[0],
            'path': alignment_before[0]
                    }

    return var_dtw



# MULTIVARIATE_ALIGNMENT
def multivariate_alignment(var_dtw):
    weights = [var_dtw[i]['ratio'] for i in var_dtw.keys()]
    paths = [var_dtw[i]['path'] for i in var_dtw.keys()]

    vals = []
    for i in range(len(paths)):
        a = pd.DataFrame(paths[i], columns = ['init', 'transf'])
        vals.extend(a['init'].unique())
    vals = np.unique(vals)

    transform_data = []
    for j in vals :
        val = 0
        for i in range(len(paths)):
            df = pd.DataFrame(paths[i], columns = ['init', 'transf'])
            val += df[df['init'] == j]['transf'].mean()*weights[i]
        val = round(val/sum(weights))
        transform_data.append(val)

    return transform_data


# MULTIVARIATE_ALIGNMENT_V2
def multivariate_alignment_v2(df1, df2, global_constraint='itakura', sakoe_chiba_radius=None, 
                              itakura_max_slope=None): 

    path = dtw_path(df1.values, df2.values, global_constraint=global_constraint, 
                           sakoe_chiba_radius=sakoe_chiba_radius, itakura_max_slope=itakura_max_slope)
    
    premiers_elements = [t[0] for t in list(path)[0]]
    seconds_elements = [t[1] for t in list(path)[0]]

    return premiers_elements, seconds_elements