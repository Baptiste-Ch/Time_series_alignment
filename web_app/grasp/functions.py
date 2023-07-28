import json
import sqlite3
from flask import request
from grasp.models import VerifData, CounterData
from grasp import db

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dtaidistance import dtw
from tslearn.metrics import dtw_path
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter, find_peaks



#############################################
##        FONCTIONS DE L APPLICATION       ##
#############################################

#READ_COUNTER
def read_counter():
    counter_entry = CounterData.query.first()
    return counter_entry.counter if counter_entry else 0


#INCREMENT_COUNTER
def increment_counter():
    counter_entry = CounterData.query.first()
    if counter_entry:
        counter_entry.counter += 1
    else:
        new_counter_entry = CounterData(counter=1)
        db.session.add(new_counter_entry)
    db.session.commit()


# COPY_TABLES
def copy_tables(source_db, target_db):
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
def overall_plot(database):
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
        selected_variable = request.form.get('variable-dropdown1') or list(shared_columns)[0]
        
        fig = make_subplots(rows=len(table_names), cols=1, shared_xaxes=True, vertical_spacing=0)

        for i, table_name in enumerate(table_names):
            query = f"SELECT {columns[1][1]}, {selected_variable} FROM {table_name}"
            data = conn.execute(query).fetchall()

            x_vals = [row[0] for row in data]
            y_vals = [row[1] for row in data]

            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=table_name), row=i+1, col=1)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig_json = json.dumps(fig.to_dict())

        return fig_json, dropdown_options


# FOCUS_PLOT
def focus_plot(init_database, step, global_constraint=None, sakoe_chiba_radius=None, 
               itakura_max_slope=None, align=None, dropdown_value=None):
    
    # Connection à la base de données initiale
    conn1 = sqlite3.connect(init_database)
    cursor = conn1.cursor()
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' LIMIT 2 OFFSET {step}")
    table_names = [row[0] for row in cursor.fetchall()]

    dfs = []
    for table_name in table_names:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn1)
        dfs.append(df)
    df1 = dfs[0]
    df2 = dfs[1]
    
    # Creation de variables
    fig_json = None
    dropdown_options = None
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
                    (itakura_max_slope == db_record.itakura_max_slope):
                    
                    var1_dtw = json.loads(db_record.var1_dtw)
                    var2_dtw = json.loads(db_record.var2_dtw)

                    selected_variable = request.form.get('variable-dropdown2')

                # SI JE CHANGE DE PARAM
                else :  
                    var1_dtw, var2_dtw = multivariate_alignment_v2(df1, df2, global_constraint=global_constraint, 
                                                                   sakoe_chiba_radius=sakoe_chiba_radius, 
                                                                   itakura_max_slope=itakura_max_slope)

                    db_record.global_constraint = global_constraint
                    db_record.sakoe_chiba_radius = sakoe_chiba_radius
                    db_record.itakura_max_slope = itakura_max_slope
                    db_record.var1_dtw = json.dumps(var1_dtw)
                    db_record.var2_dtw = json.dumps(var2_dtw)
                    db.session.commit()

                    if dropdown_value :
                        selected_variable = dropdown_value
                    else :
                        selected_variable = list(shared_columns)[0]                    


            # SI C'EST LA PREMIERE FOIS
            else :   
                var1_dtw, var2_dtw = multivariate_alignment_v2(df1, df2, global_constraint=global_constraint, 
                                                               sakoe_chiba_radius=sakoe_chiba_radius, 
                                                               itakura_max_slope=itakura_max_slope)

                new_record = VerifData(
                    global_constraint=global_constraint,
                    sakoe_chiba_radius=sakoe_chiba_radius,
                    itakura_max_slope=itakura_max_slope,
                    var1_dtw=json.dumps(var1_dtw),
                    var2_dtw=json.dumps(var2_dtw))
                db.session.add(new_record)             
                db.session.commit()

                selected_variable = dropdown_value

            df1 = df1.iloc[var1_dtw].reset_index(drop=True)
            df2 = df2.iloc[var2_dtw].reset_index(drop=True)
            df2.iloc[:, 1] = df1.iloc[:, 1]

            conn3 = sqlite3.connect('temporary_data.db')
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
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig_json = json.dumps(fig.to_dict())

        return fig_json, dropdown_options


# FORWARD_COUNTER
def forward_counter(counter) :
    counter += 1
    return counter

# BACK_COUNTER
def back_counter(counter) :
    counter -= 1
    return counter




############################################
##        FONCTIONS DE L ALGORITHME       ##
############################################


# EXTRACT_ELEMENTS
def extract_elements(liste_de_tuples):
    premiers_elements = [t[0] for t in liste_de_tuples]
    seconds_elements = [t[1] for t in liste_de_tuples]
    return premiers_elements, seconds_elements


# UNIVARIATE_ALIGNMENT
def univariate_alignment(df1, df2, global_constraint='itakura', sakoe_chiba_radius=None,
                         itakura_max_slope=None):
    
    var_names = df1.iloc[:, 1:].columns.tolist()
    print('DF1')
    print(df1)
    print('DF2')
    print(df2)

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

    path = list(dtw_path(df1.values, df2.values, global_constraint=global_constraint, 
                           sakoe_chiba_radius=sakoe_chiba_radius, itakura_max_slope=itakura_max_slope))[0]
    
    premiers_elements = [t[0] for t in path]
    seconds_elements = [t[1] for t in path]

    return premiers_elements, seconds_elements