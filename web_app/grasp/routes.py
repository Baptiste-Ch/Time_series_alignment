import os
import numpy as np
import sys
import ast
from flask import render_template, url_for, redirect, request, session, jsonify, send_file, current_app
from grasp import app, db
from grasp.models import InputData, CounterData, VerifData, DropdownData1, DropdownData2, HistoricData

import pandas as pd
import sqlite3
import json


sys.path.append(app.root_path)

from functions import overall_plot, focus_plot, focus_plot2, read_counter, increment_counter, reorder_tables, text_to_nan, decrease_counter, interpolation, text_to_float32



#--------------- HOME ------------------ #
# Define a route for the homepage ("/" and "/home" URLs) using the @app.route decorator.
# When a user accesses these URLs, the 'home' function is called.





#-------------- UPLOAD ---------------#
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle file upload, data processing, and database creation."""

    # Reset the 'loaded' session variable.
    session['loaded'] = None

    # Get the uploaded file from the request.
    file = request.files.get('file')

    # Define the file path to save the uploaded file.
    file_path = os.path.join(app.root_path, 'upload', file.filename)

    # Save the uploaded file to the defined file path.
    file.save(file_path)

    # Store the uploaded file path in the session.
    session['uploaded_file_path'] = file_path

    # Read the Excel file into a Pandas DataFrame and process it.
    df = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
    for sheet_name, item in df.items():
        item.iloc[1:] = item.iloc[1:].applymap(text_to_nan)
        numeric_columns = item.select_dtypes(include=[np.number]).columns
        item[numeric_columns] = item[numeric_columns].astype(np.float32)


    # Create SQLite databases and connect to them.
    app_path = os.path.dirname(app.root_path)
    for db_file in ['init_data.db', 'modified_data.db']:
        path_to_delete = os.path.join(app_path, db_file)
        if os.path.exists(path_to_delete):
            os.remove(path_to_delete)
    conn_1 = sqlite3.connect('init_data.db')
    conn_2 = sqlite3.connect('modified_data.db')

    # Loop over the sheets in the Excel file and save them to the databases.
    for conn in [conn_1, conn_2]:
        for sheet_name, sheet_data in df.items():
            sheet_data.to_sql(sheet_name, conn, if_exists='replace')
        conn.close()

    # Redirect to the 'displays' route.
    return redirect(url_for('displays'))


#---------------UPDATE PLOT DATA -----------------
@app.route('/update_plot_data', methods=['GET', 'POST'])
def update_plot_data():
    init_json = {}  # Use an empty dictionary as a placeholder
    init_dropdown = []  # Use an empty list as a placeholder
    modified_json = {}  # Use an empty dictionary as a placeholder
    modified_dropdown = []  # Use an empty list as a placeholder
    step = 0
    focus_json = {}  # Use an empty dictionary as a placeholder
    focus_dropdown = []  # Use an empty list as a placeholder

    conn_1 = sqlite3.connect('init_data.db')
    conn_2 = sqlite3.connect('modified_data.db')
    cursor1 = conn_1.cursor()
    cursor2 = conn_2.cursor()
    cursor1.execute("SELECT name FROM sqlite_master WHERE type='table';")
    cursor2.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names1 = cursor1.fetchall()
    table_names2 = cursor2.fetchall()
    print(table_names1, table_names2)

    if table_names1 and table_names2:
        print('IM INSIDE')
        # Generate JSON data and dropdown options for the initial state.
        init_json, init_dropdown = overall_plot('init_data.db')

        # Generate JSON data and dropdown options for the modified state.
        modified_json, modified_dropdown = overall_plot('modified_data.db')

        # Initialize the step to 0 and generate JSON data and dropdown options for the focus plot.
        step = 0
        focus_json, focus_dropdown = focus_plot('init_data.db', step)

        # Retrieve or create a DropdownData2 object and update its dropdown value.
        dropdown_data = DropdownData2.query.first()
        if dropdown_data:
            dropdown_data.dropdown_value = focus_dropdown[0]['label']
            print(focus_dropdown[0]['label'])
        else:
            new_dropdown_data = DropdownData2(dropdown_value=focus_dropdown[0]['label'])
            db.session.add(new_dropdown_data)
        db.session.commit()
    conn_1.close()
    conn_2.close()
    
    return jsonify({"init_json" : init_json, 
            "init_dropdown" : init_dropdown, 
            "modified_json" : modified_json, 
            "modified_dropdown" : modified_dropdown, 
            "focus_json" : focus_json, 
            "focus_dropdown" : focus_dropdown })


#-----------------DISPLAYS-------------------#
@app.route("/")
@app.route("/displays", methods=['GET', 'POST'])
def displays():
    """Goal: Generate data for visualization plots and update dropdown value."""

    init_json = {}  # Use an empty dictionary as a placeholder
    init_dropdown = []  # Use an empty list as a placeholder
    modified_json = {}  # Use an empty dictionary as a placeholder
    modified_dropdown = []  # Use an empty list as a placeholder
    step = 0
    focus_json = {}  # Use an empty dictionary as a placeholder
    focus_dropdown = []  # Use an empty list as a placeholder

    # Render the 'displays.html' template with the generated data.
    return render_template('displays.html', init_json=init_json, init_dropdown=init_dropdown,
                           modified_json=modified_json, modified_dropdown=modified_dropdown,
                           focus_json=focus_json, focus_dropdown=focus_dropdown)


#-------------- RESET -----------------#
@app.route('/displays/reset', methods=['POST'])
def reset():
    # Goal: Reset the alignment data and generate JSON data for the reset focus plot.

    # Reset the 'align' attribute of 'InputData' in the database, if available.
    input_data = InputData.query.first()
    if input_data:
        input_data.align = None
        db.session.commit()

    # Read the current step from a counter.
    step = read_counter()
    print('RESET')

    # Retrieve the dropdown value from the database.
    dropdown_data = DropdownData2.query.first()
    dropdown_value = None
    if dropdown_data:
        dropdown_value = dropdown_data.dropdown_value

    # Generate JSON data for the reset focus plot.
    data = json.loads(request.data)
    choice = data.get('choice')

    if choice == 0 :
        reset_json, _ = focus_plot('init_data.db', step=step, dropdown_value=dropdown_value)
    if choice == 1 :
        reset_json, _ = focus_plot2('init_data.db', step=step, dropdown_value=dropdown_value)
    
    # Return the reset JSON data.
    return {'reset_json': reset_json}


#--------------- DISPLAYS - FIGURE_INIT -----------#
@app.route("/displays/figure_init", methods=['POST'])
def figure_init():
    # Goal: Generate and return JSON data for the initial state visualization plot.

    # Generate JSON data for the initial state visualization plot.
    init_json, _ = overall_plot('init_data.db')

    # Return the generated JSON data.
    return jsonify(init_json=init_json)


#--------------- DISPLAYS - FIGURE_MODIFIED -----------#
@app.route("/displays/figure_modified", methods=['POST'])
def figure_modified():
    # Goal: Generate JSON data for the modified state visualization plot and handle dropdown value updates.

    # Generate JSON data for the modified state visualization plot.
    print('FIGURE MODIFIED')
    modified_json, _ = overall_plot('modified_data.db')

    # Retrieve the selected dropdown value from the request.
    dropdown_value = request.form.get('variable-dropdown1', None)

    # Retrieve or create a DropdownData1 object and update its dropdown value.
    dropdown_data = DropdownData1.query.first()
    if dropdown_value:
        if not dropdown_data:
            new_dropdown_data = DropdownData1(dropdown_value=dropdown_value)
            db.session.add(new_dropdown_data)
        else:
            dropdown_data.dropdown_value = dropdown_value
        db.session.commit()

    # Return the generated JSON data.
    return jsonify(modified_json=modified_json)


#--------------- DISPLAYS - FIGURE_FOCUS -----------#
@app.route("/displays/figure_focus", methods=['GET', 'POST'])
def figure_focus():
    # Goal: Generate JSON data for the focus plot and manage dropdown value updates.

    # Initialize variables to None.
    align = None
    global_constraint = None
    sakoe_chiba_radius = None
    itakura_max_slope = None
    selected_vars = None

    # Retrieve the dropdown value from the request or the database.
    dropdown_data = DropdownData2.query.first()
    if request.args.get('dropdown_value'):
        dropdown_value = request.args.get('dropdown_value')
    else:
        dropdown_value = request.form.get('variable-dropdown2', None)
        if not dropdown_value:
            if dropdown_data:
                dropdown_value = dropdown_data.dropdown_value

    # Update or create a DropdownData2 object and set its dropdown value.
    if dropdown_value:
        if not dropdown_data:
            new_dropdown_data = DropdownData2(dropdown_value=dropdown_value)
            db.session.add(new_dropdown_data)
        else:
            dropdown_data.dropdown_value = dropdown_value
        db.session.commit()

    # Check if alignment data exists in the database and retrieve it.
    with current_app.app_context():
        exists = db.session.query(db.exists().where(InputData.align.isnot(None))).scalar()
        if exists:
            input_data = InputData.query.first()
            align = input_data.align
            global_constraint = input_data.global_constraint
            sakoe_chiba_radius = input_data.sakoe_chiba_radius
            itakura_max_slope = input_data.itakura_max_slope
            selected_vars = input_data.selected_vars

    # Read the current step from a counter.
    step = read_counter()

    # Generate JSON data for the focus plot.
    focus_json, _ = focus_plot('init_data.db', step=step, global_constraint=global_constraint,
                               sakoe_chiba_radius=sakoe_chiba_radius, itakura_max_slope=itakura_max_slope,
                               align=align, dropdown_value=dropdown_value, selected_vars=selected_vars)

    # Return the generated focus JSON data.
    return {'focus_json': focus_json}



#--------------- DISPLAYS - FIGURE_FOCUS2 -----------#
@app.route("/displays/figure_focus2", methods=['GET', 'POST'])
def figure_focus2():
    # Goal: Generate JSON data for an alternative focus plot and manage dropdown value updates.

    # Initialize variables to None.
    align = None
    global_constraint = None
    sakoe_chiba_radius = None
    itakura_max_slope = None
    selected_vars = None

    # Retrieve the dropdown value from the request or the database.
    dropdown_data = DropdownData2.query.first()
    if request.args.get('dropdown_value'):
        dropdown_value = request.args.get('dropdown_value')
    else:
        dropdown_value = request.form.get('variable-dropdown2', None)
        if not dropdown_value:
            if dropdown_data:
                dropdown_value = dropdown_data.dropdown_value

    # Update or create a DropdownData2 object and set its dropdown value.
    if dropdown_value:
        if not dropdown_data:
            new_dropdown_data = DropdownData2(dropdown_value=dropdown_value)
            db.session.add(new_dropdown_data)
        else:
            dropdown_data.dropdown_value = dropdown_value
        db.session.commit()

    # Check if alignment data exists in the database and retrieve it.
    with current_app.app_context():
        exists = db.session.query(db.exists().where(InputData.align.isnot(None))).scalar()
        if exists:
            input_data = InputData.query.first()
            align = input_data.align
            global_constraint = input_data.global_constraint
            sakoe_chiba_radius = input_data.sakoe_chiba_radius
            itakura_max_slope = input_data.itakura_max_slope
            selected_vars = input_data.selected_vars

    # Read the current step from a counter.
    step = read_counter()

    # Generate JSON data for the alternative focus plot.
    focus_json, _ = focus_plot2('init_data.db', step=step, global_constraint=global_constraint,
                               sakoe_chiba_radius=sakoe_chiba_radius, itakura_max_slope=itakura_max_slope,
                               align=align, dropdown_value=dropdown_value, selected_vars=selected_vars)

    # Return the generated focus JSON data.
    return {'focus_json': focus_json}



#------------NEXT DATA---------------------#
@app.route('/next_data', methods=['GET', 'POST'])
def next_data():
    # Goal: Handle the retrieval and processing of data for the 'next_data' route.

    print('NEXT DATA')
    
    # Initialize dropdown values to None.
    dropdown_value2 = None
    dropdown_value1 = None

    # Read the current step from a counter or set it to 0 if not available.
    step = read_counter()
    if not step:
        step = 0

    # Delete all records from the 'VerifData' table in the database.
    db.session.query(VerifData).delete()

    # Connect to two SQLite databases, 'temporary_data.db' and 'modified_data.db', and retrieve table names.
    conn1 = sqlite3.connect('temporary_data.db')
    conn2 = sqlite3.connect('modified_data.db')
    cursor1 = conn1.cursor()
    cursor2 = conn2.cursor()
    cursor1.execute("SELECT name FROM sqlite_master WHERE type='table';")
    cursor2.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names1 = [row[0] for row in cursor1.fetchall()]

    # Read data from tables in 'temporary_data.db'.
    df1 = pd.read_sql(f'SELECT * FROM {table_names1[0]}', conn1)
    df2 = pd.read_sql(f'SELECT * FROM {table_names1[1]}', conn1)
    inter2 = df2

    # Fetch historical data from 'HistoricData' table and perform data transformations.
    history = HistoricData.query.limit(step).all()
    if history:
        for i in reversed(range(len(history))):
            lst_elem = ast.literal_eval(history[i].var2_dtw)
            lgth_last_elem = history[i].lgth_last_elem
            inter2 = interpolation(lgth_last_elem, inter2, lst_elem, history[i].min_depth, history[i].max_depth)
            inter2 = inter2.iloc[lst_elem].reset_index(drop=True)

    # Drop and replace a table in 'modified_data.db' with the transformed data.
    cursor2.execute(f"DROP TABLE IF EXISTS {table_names1[1]}")
    inter2.to_sql(table_names1[1], conn2, index=False)
    conn2.commit()
    conn2.close()
    conn1.close()

    # Remove the 'temporary_data.db' file if it exists.
    app_path = os.path.dirname(app.root_path)
    if os.path.exists(os.path.join(app_path, 'temporary_data.db')):
        os.remove(os.path.join(app_path, 'temporary_data.db'))
    

    # Reorder tables in 'init_data.db' and 'modified_data.db'.
    reorder_tables('init_data.db', 'modified_data.db')

    # Increment the counter and reset dropdown values based on database entries.
    counter = CounterData.query.first()
    if not counter or (counter.counter < counter.max_counter-2):
        increment_counter('init_data.db')
    input_data = InputData.query.first()
    if input_data:
        input_data.align = None
    dropdown_data2 = DropdownData2.query.first()
    if dropdown_data2:
        dropdown_value2 = dropdown_data2.dropdown_value
        db.session.commit()
    dropdwon_data1 = DropdownData1.query.first()
    if dropdwon_data1:
        dropdown_value1 = dropdwon_data1.dropdown_value
        db.session.commit()

    # Read the step again.
    step = read_counter()

    # Generate JSON data and dropdown options for visualization plots.
    modified_json, modified_dropdown = overall_plot('modified_data.db', dropdown_value=dropdown_value1)
    focus_json, focus_dropdown = focus_plot('init_data.db', step, dropdown_value=dropdown_value2)
    response_data = {
        'modified_json': modified_json,
        'focus_json': focus_json,
        'modified_dropdown': modified_dropdown,
        'focus_dropdown': focus_dropdown
    }

    # Return the response data as JSON.
    return jsonify(response_data)



#------------NEXT DATA2---------------------#
@app.route('/next_data2', methods=['GET', 'POST'])
def next_data2():
    # Goal: Handle the retrieval and processing of data for the 'next_data2' route.

    print('NEXT DATA2')
    
    # Initialize dropdown values to None.
    dropdown_value2 = None
    dropdown_value1 = None

    # Read the current step from a counter or set it to 0 if not available.
    step = read_counter()
    if not step:
        step = 0

    # Delete all records from the 'VerifData' table in the database.
    db.session.query(VerifData).delete()

    # Connect to two SQLite databases, 'temporary_data.db' and 'modified_data.db', and retrieve table names.
    conn1 = sqlite3.connect('temporary_data.db')
    conn2 = sqlite3.connect('modified_data.db')
    cursor1 = conn1.cursor()
    cursor2 = conn2.cursor()
    cursor1.execute("SELECT name FROM sqlite_master WHERE type='table';")
    cursor2.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names1 = [row[0] for row in cursor1.fetchall()]

    # Read data from a table in 'temporary_data.db'.
    df2 = pd.read_sql(f'SELECT * FROM {table_names1[1]}', conn1)

    # Drop and replace a table in 'modified_data.db' with the transformed data.
    cursor2.execute(f"DROP TABLE IF EXISTS {table_names1[1]}")
    df2.to_sql(table_names1[1], conn2, index=False)
    conn2.commit()
    conn2.close()

    # Remove the 'temporary_data.db' file if it exists.
    conn1.close()
    app_path = os.path.dirname(app.root_path)
    if os.path.exists(os.path.join(app_path, 'temporary_data.db')):
        os.remove(os.path.join(app_path, 'temporary_data.db'))
    

    # Reorder tables in 'init_data.db' and 'modified_data.db'.
    reorder_tables('init_data.db', 'modified_data.db')

    # Increment the counter and reset dropdown values based on database entries.
    counter = CounterData.query.first()
    if not counter or (counter.counter < counter.max_counter-2):
        increment_counter('init_data.db')
    input_data = InputData.query.first()
    if input_data:
        input_data.align = None
    dropdown_data2 = DropdownData2.query.first()
    if dropdown_data2:
        dropdown_value2 = dropdown_data2.dropdown_value
        db.session.commit()
    dropdwon_data1 = DropdownData1.query.first()
    if dropdwon_data1:
        dropdown_value1 = dropdwon_data1.dropdown_value
        db.session.commit()

    # Read the step again.
    step = read_counter()

    # Generate JSON data and dropdown options for visualization plots.
    modified_json, modified_dropdown = overall_plot('modified_data.db', dropdown_value=dropdown_value1)
    focus_json, focus_dropdown = focus_plot2('init_data.db', step, dropdown_value=dropdown_value2)
    response_data = {
        'modified_json': modified_json,
        'focus_json': focus_json,
        'modified_dropdown': modified_dropdown,
        'focus_dropdown': focus_dropdown
    }

    # Return the response data as JSON.
    return jsonify(response_data)


#----------------- BACK DATA -------------------#
@app.route('/back_data', methods=['GET', 'POST'])
def back_data():
    # Goal: Handle the data and session cleanup when returning to a previous state.

    # Retrieve the selected dropdown value from the database, if available.
    dropdown_data = DropdownData2.query.first()
    dropdown_value = None
    if dropdown_data:
        dropdown_value = dropdown_data.dropdown_value
        print('SELECTED VARS:', dropdown_value)
        db.session.commit()

    # Reset the 'align' attribute of 'InputData' in the database, if available.
    input_data = InputData.query.first()
    if input_data:
        input_data.align = None
        db.session.commit()

    # Delete all records from the 'VerifData' table and commit changes.
    db.session.query(VerifData).delete()
    db.session.commit()

    # Decrease the counter value if it's greater than 0.
    counter = CounterData.query.first()
    if counter and counter.counter > 0:
        decrease_counter()

    # Remove the 'temporary_data.db' file if it exists.
    app_path = os.path.dirname(app.root_path)
    if os.path.exists(os.path.join(app_path, 'temporary_data.db')):
        os.remove(os.path.join(app_path, 'temporary_data.db'))
    

    data = json.loads(request.data)
    choice = data.get('choice')

    if choice == 0 :
        which_focus = 'figure_focus'
    if choice == 1 :
        which_focus = 'figure_focus2'

    # Redirect to the 'figure_focus' route with the selected dropdown value.
    return redirect(url_for(which_focus, dropdown_value=dropdown_value))


#--------------- STORE DATA ------------------------#
@app.route('/store_data', methods=['POST'])
def store_data():
    # Goal: Receive and store alignment and configuration data in the database.

    # Parse JSON data from the request.
    data = request.get_json()
    align = data.get('align')
    global_constraint = data.get('global_constraint')
    sakoe_chiba_radius = data.get('sakoe_chiba_radius')
    itakura_max_slope = data.get('itakura_max_slope')
    dropdown_value = data.get('dropdown_value')
    selected_vars = str(data.get('selected_vars'))

    # Retrieve the existing InputData row or create a new one.
    input_data = InputData.query.first()
    if input_data:
        # Update the existing row with the new data.
        input_data.align = align
        input_data.global_constraint = global_constraint
        input_data.sakoe_chiba_radius = sakoe_chiba_radius
        input_data.itakura_max_slope = itakura_max_slope
        input_data.dropdown_value = dropdown_value
        input_data.selected_vars = selected_vars
    else:
        # Create a new row with the received data.
        input_data = InputData(
            align=align,
            global_constraint=global_constraint,
            sakoe_chiba_radius=sakoe_chiba_radius,
            itakura_max_slope=itakura_max_slope,
            dropdown_value=dropdown_value,
            selected_vars=selected_vars
        )
        db.session.add(input_data)
    
    # Commit the changes to the database.
    db.session.commit()
    
    # Return a success response.
    return jsonify({'success': True})


#---------------- DOWNLOAD ---------------------#
@app.route('/download', methods=['GET'])
def download():
    # Goal: Generate an Excel file containing data from SQLite tables and allow it to be downloaded.

    # Get the absolute path of the application directory.
    app_path = os.path.dirname(os.path.abspath(__file__))

    # Define the output Excel file path in the 'upload' directory.
    output_excel_path = os.path.join(app_path, 'upload', 'aligned_data.xlsx') 

    # Connect to the SQLite database and fetch table names.
    conn = sqlite3.connect('modified_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [row[0] for row in cursor.fetchall()]

    # Create an Excel writer for the output file.
    writer = pd.ExcelWriter(output_excel_path, engine='openpyxl')

    # Loop through tables and fetch data from the database.
    for table_name in table_names:
        df = pd.read_sql_query(f'SELECT * FROM {table_name}', conn)
        df.to_excel(writer, sheet_name=table_name, index=False)

    # Save and close the Excel writer and database connection.
    writer.save()
    conn.close()

    # Return the generated Excel file as an attachment for download.
    return send_file(output_excel_path, as_attachment=True)


#-----------------OTHER PAGES--------------------#

@app.route("/about", methods=['GET', 'POST'])
def about():
    return render_template("about.html")


@app.route("/documentation", methods=['GET', 'POST'])
def documentation():
    return render_template("documentation.html")

