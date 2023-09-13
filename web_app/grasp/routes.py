import os
import sys
import ast
import secrets
from PIL import Image, ImageOps
from flask import render_template, url_for, flash, redirect, request, session, jsonify, send_file, current_app
from grasp import app, db, bcrypt
from grasp.models import User, InputData, CounterData, VerifData, DropdownData1, DropdownData2, HistoricData
from grasp.forms import RegistrationForm, LoginForm, UpdateAccountForm
from flask_login import login_user, current_user, logout_user, login_required


import pandas as pd
import numpy as np
import time
import json

import sqlite3


sys.path.append(app.root_path)

from functions import overall_plot, focus_plot, focus_plot2, copy_tables, read_counter, increment_counter, reorder_tables, text_to_nan, decrease_counter, align_from_history, multivariate_alignment_v2, interpolation



#--------------- HOME ------------------ #
@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

#---------------- DOWNLOAD ---------------------#
@app.route('/download', methods=['GET'])
def download():
    app_path = os.path.dirname(os.path.abspath(__file__))

    output_excel_path = os.path.join(app_path, 'upload', 'aligned_data.xlsx') 

    conn = sqlite3.connect('modified_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [row[0] for row in cursor.fetchall()]

    writer = pd.ExcelWriter(output_excel_path, engine='openpyxl')

    for table_name in table_names:
        df = pd.read_sql_query(f'SELECT * FROM {table_name}', conn)
        df.to_excel(writer, sheet_name=table_name, index=False)

    writer.save()
    conn.close()

    return send_file(output_excel_path, as_attachment=True)



#---------------- TRANSFER TABLES -------------#
@app.route('/transfer_tables', methods=['POST'])
def transfer_tables():
    try:
        conn1 = sqlite3.connect('temporary_data.db')
        conn2 = sqlite3.connect('modified_data.db')
        cursor1 = conn1.cursor()
        cursor2 = conn2.cursor()
        cursor1.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [row[0] for row in cursor1.fetchall()]

        for table_name in table_names:
            cursor1.execute(f"SELECT * FROM {table_name}")
            table_data = cursor1.fetchall()
            cursor2.execute(f"DELETE FROM {table_name}")
            cursor2.executemany(f"INSERT INTO {table_name} VALUES (?, ?)", table_data)
            print(f"Table: {table_name}")
            for row in table_data:
                print(row)
        conn2.commit()
        conn1.close()
        conn2.close()


        return jsonify({"message": "Tables transferred successfully!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



#----------------- BACK DATA -------------------#
@app.route('/back_data', methods=['GET', 'POST'])
def back_data():
    dropdown_data = DropdownData2.query.first()
    dropdown_value = None
    if dropdown_data :
        dropdown_value = dropdown_data.dropdown_value
        print('SELECTED VARS :', dropdown_value)
        db.session.commit()

    input_data = InputData.query.first()
    if input_data :
        input_data.align = None
        db.session.commit()

    db.session.query(VerifData).delete()
    db.session.commit()

    counter = CounterData.query.first()
    if counter :
        if counter.counter > 0 :
            decrease_counter()
        print(counter.counter)

    conn1 = sqlite3.connect('temporary_data.db')
    app_path = os.path.dirname(app.root_path)
    if os.path.exists(os.path.join(app_path, 'temporary_data.db')):
        os.remove(os.path.join(app_path, 'temporary_data.db'))
    conn1.close()
    return redirect(url_for('figure_focus', dropdown_value=dropdown_value))



#------------NEXT DATA---------------------#
@app.route('/next_data', methods=['GET', 'POST'])
def next_data():
    print('NEXT DATA')
    dropdown_value2 = None
    dropdown_value1 = None

    step = read_counter()
    if not step:
        step = 0

    db.session.query(VerifData).delete()


    #Transformer la données tout juste alignée en fonction des alignements précédents
    conn1 = sqlite3.connect('temporary_data.db')
    conn2 = sqlite3.connect('modified_data.db')
    cursor1 = conn1.cursor()
    cursor2 = conn2.cursor()
    cursor1.execute("SELECT name FROM sqlite_master WHERE type='table';")
    cursor2.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names1 = [row[0] for row in cursor1.fetchall()]

    df1 = pd.read_sql(f'SELECT * FROM {table_names1[0]}', conn1)
    df2 = pd.read_sql(f'SELECT * FROM {table_names1[1]}', conn1)
    inter_empty = True

    history = HistoricData.query.limit(step).all()

    if history :
        for i in reversed(range(len(history))):
            lst_elem = ast.literal_eval(history[i].var2_dtw)

            lgth_last_elem = history[i].lgth_last_elem
            inter1 = interpolation(lgth_last_elem, df1)
            inter2 = interpolation(lgth_last_elem, df2)          
            
            inter1 = inter1.iloc[lst_elem].reset_index(drop=True)
            inter2 = inter2.iloc[lst_elem].reset_index(drop=True)

            inter_empty = False
            
    if inter_empty :
        inter2 = df2

    cursor2.execute(f"DROP TABLE IF EXISTS {table_names1[1]}")
    inter2.to_sql(table_names1[1], conn2, index=False)
    conn2.commit()
    conn2.close()

    app_path = os.path.dirname(app.root_path)
    if os.path.exists(os.path.join(app_path, 'temporary_data.db')):
        os.remove(os.path.join(app_path, 'temporary_data.db'))
    conn1.close()



    #copy_tables('transformed_data.db', 'modified_data.db')
    reorder_tables('init_data.db', 'modified_data.db')

    counter = CounterData.query.first()
    if not counter or (counter.counter < counter.max_counter-2) :  
        increment_counter('init_data.db')
        input_data = InputData.query.first()
        if input_data :
            input_data.align = None
        dropdown_data2 = DropdownData2.query.first()
        if dropdown_data2 :
            dropdown_value2 = dropdown_data2.dropdown_value
            db.session.commit()
        dropdwon_data1 = DropdownData1.query.first()
        if dropdwon_data1 :
            dropdown_value1 = dropdwon_data1.dropdown_value
            db.session.commit()

    step = read_counter()


    modified_json, modified_dropdown = overall_plot('modified_data.db', dropdown_value=dropdown_value1)
    focus_json, focus_dropdown = focus_plot('init_data.db', step, dropdown_value=dropdown_value2)
    response_data = {
        'modified_json': modified_json,
        'focus_json': focus_json,
        'modified_dropdown': modified_dropdown,
        'focus_dropdown': focus_dropdown
    }

    return jsonify(response_data)


#------------NEXT DATA---------------------#
@app.route('/next_data2', methods=['GET', 'POST'])
def next_data2():
    print('NEXT DATA2')
    dropdown_value2 = None
    dropdown_value1 = None

    step = read_counter()
    if not step:
        step = 0

    db.session.query(VerifData).delete()


    #Transformer la données tout juste alignée en fonction des alignements précédents
    conn1 = sqlite3.connect('temporary_data.db')
    conn2 = sqlite3.connect('modified_data.db')
    cursor1 = conn1.cursor()
    cursor2 = conn2.cursor()
    cursor1.execute("SELECT name FROM sqlite_master WHERE type='table';")
    cursor2.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names1 = [row[0] for row in cursor1.fetchall()]

    df1 = pd.read_sql(f'SELECT * FROM {table_names1[0]}', conn1)
    df2 = pd.read_sql(f'SELECT * FROM {table_names1[1]}', conn1)


    cursor2.execute(f"DROP TABLE IF EXISTS {table_names1[1]}")
    df2.to_sql(table_names1[1], conn2, index=False)
    conn2.commit()
    conn2.close()

    app_path = os.path.dirname(app.root_path)
    if os.path.exists(os.path.join(app_path, 'temporary_data.db')):
        os.remove(os.path.join(app_path, 'temporary_data.db'))
    conn1.close()



    #copy_tables('transformed_data.db', 'modified_data.db')
    reorder_tables('init_data.db', 'modified_data.db')

    counter = CounterData.query.first()
    if not counter or (counter.counter < counter.max_counter-2) :  
        increment_counter('init_data.db')
        input_data = InputData.query.first()
        if input_data :
            input_data.align = None
        dropdown_data2 = DropdownData2.query.first()
        if dropdown_data2 :
            dropdown_value2 = dropdown_data2.dropdown_value
            db.session.commit()
        dropdwon_data1 = DropdownData1.query.first()
        if dropdwon_data1 :
            dropdown_value1 = dropdwon_data1.dropdown_value
            db.session.commit()

    step = read_counter()


    modified_json, modified_dropdown = overall_plot('modified_data.db', dropdown_value=dropdown_value1)
    focus_json, focus_dropdown = focus_plot2('init_data.db', step, dropdown_value=dropdown_value2)
    response_data = {
        'modified_json': modified_json,
        'focus_json': focus_json,
        'modified_dropdown': modified_dropdown,
        'focus_dropdown': focus_dropdown
    }

    return jsonify(response_data)



#-------------- UPLOAD ---------------#
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    session['loaded'] = None
    file = request.files.get('file')
    file_path = os.path.join(app.root_path, 'upload', file.filename)
    file.save(file_path)
    session['uploaded_file_path'] = file_path
    
    df = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')   
    for sheet_name, item in df.items():
        df[sheet_name].iloc[1:] = item.iloc[1:].applymap(text_to_nan)

    # Create a SQLite database and connect to it
    app_path = os.path.dirname(app.root_path)
    for i in ['init_data.db', 'modified_data.db'] :
        path_to_delete = os.path.join(app_path, i)
        if os.path.exists(path_to_delete):
            os.remove(path_to_delete)
    conn_1 = sqlite3.connect('init_data.db')
    conn_2 = sqlite3.connect('modified_data.db')

    # Loop over the sheets in the Excel file
    for i in [conn_1, conn_2] :
        for sheet_name, sheet_data in df.items():
            sheet_data.to_sql(sheet_name, i, if_exists='replace')
        i.close()

    return redirect(url_for('displays'))


#-----------------DISPLAYS-------------------#
@app.route("/displays", methods=['GET', 'POST'])
def displays():
    init_json, init_dropdown = overall_plot('init_data.db')
    modified_json, modified_dropdown = overall_plot('modified_data.db')

    #step = read_counter()
    step = 0
    focus_json, focus_dropdown = focus_plot('init_data.db', step)

    dropdown_data = DropdownData2.query.first()
    if dropdown_data:
        dropdown_data.dropdown_value = focus_dropdown[0]['label']
    else:
        new_dropdown_data = DropdownData2(dropdown_value=focus_dropdown[0]['label'])
        db.session.add(new_dropdown_data)
    db.session.commit()

    return render_template('displays.html', init_json=init_json, init_dropdown=init_dropdown,
                           modified_json=modified_json, modified_dropdown=modified_dropdown,
                           focus_json=focus_json, focus_dropdown=focus_dropdown)


#--------------- ALIGNMENTS ------------------ #
@app.route("/alignments", methods=['GET', 'POST'])
def alignments():    
    init_json, init_dropdown = overall_plot('init_data.db')
    modified_json, modified_dropdown = overall_plot('modified_data.db')

    step = 0
    focus_json, focus_dropdown = focus_plot('init_data.db', step)
    return render_template('alignments.html', init_json=init_json, init_dropdown=init_dropdown,
                           modified_json=modified_json, modified_dropdown=modified_dropdown,
                           focus_json=focus_json, focus_dropdown=focus_dropdown)


#-------------- RESET -----------------#
@app.route('/displays/reset', methods=['POST'])
def reset():
    input_data = InputData.query.first()
    if input_data :
        input_data.align = None
        db.session.commit()
    step = read_counter()
    print('RESET')

    dropdown_data = DropdownData2.query.first()
    dropdown_value = None
    if dropdown_data :
        dropdown_value = dropdown_data.dropdown_value

    reset_json, _ = focus_plot('init_data.db', step=step, dropdown_value=dropdown_value)
    return {'reset_json': reset_json}


#--------------- DISPLAYS - FIGURE_INIT -----------#
@app.route("/displays/figure_init", methods=['POST'])
def figure_init():
    init_json, _ = overall_plot('init_data.db')
    return jsonify(init_json=init_json)


#--------------- DISPLAYS - FIGURE_MODIFIED -----------#
@app.route("/displays/figure_modified", methods=['POST'])
def figure_modified():
    modified_json, _ = overall_plot('modified_data.db')

    dropdown_data = DropdownData1.query.first()
    dropdown_value = request.form.get('variable-dropdown1', None)
    if dropdown_value :
        if not dropdown_data :
            new_dropdown_data = DropdownData1(dropdown_value=dropdown_value)
            db.session.add(new_dropdown_data)
        else :
            dropdown_data.dropdown_value = dropdown_value
        db.session.commit()

    return jsonify(modified_json=modified_json)


#--------------- DISPLAYS - FIGURE_FOCUS -----------#
@app.route("/displays/figure_focus", methods=['GET', 'POST'])
def figure_focus():
    print('FIGURE FOCUS')
    align=None
    global_constraint=None
    sakoe_chiba_radius=None
    itakura_max_slope=None
    selected_vars = None

    dropdown_data = DropdownData2.query.first()
    if request.args.get('dropdown_value') :
        dropdown_value = request.args.get('dropdown_value')
    else :
        dropdown_value = request.form.get('variable-dropdown2', None)
        if not dropdown_value :
           if dropdown_data :
               dropdown_value = dropdown_data.dropdown_value         
    
    if dropdown_value :
        if not dropdown_data:
            new_dropdown_data = DropdownData2(dropdown_value=dropdown_value)
            db.session.add(new_dropdown_data)
        else:
            dropdown_data.dropdown_value = dropdown_value
        db.session.commit()
    

    with current_app.app_context():
        exists = db.session.query(db.exists().where(InputData.align.isnot(None))).scalar()
        if exists:
            input_data = InputData.query.first()
            align = input_data.align
            global_constraint = input_data.global_constraint
            sakoe_chiba_radius = input_data.sakoe_chiba_radius
            itakura_max_slope = input_data.itakura_max_slope
            selected_vars = input_data.selected_vars

    step = read_counter()
    focus_json, _ = focus_plot('init_data.db', step=step, global_constraint=global_constraint, 
                               sakoe_chiba_radius=sakoe_chiba_radius, itakura_max_slope=itakura_max_slope, 
                               align=align, dropdown_value=dropdown_value, selected_vars=selected_vars)
    return {'focus_json': focus_json}


#--------------- DISPLAYS - FIGURE_FOCUS2 -----------#
@app.route("/displays/figure_focus2", methods=['GET', 'POST'])
def figure_focus2():
    print('FIGURE FOCUS 2')
    align=None
    global_constraint=None
    sakoe_chiba_radius=None
    itakura_max_slope=None
    selected_vars = None

    dropdown_data = DropdownData2.query.first()
    if request.args.get('dropdown_value') :
        dropdown_value = request.args.get('dropdown_value')
    else :
        dropdown_value = request.form.get('variable-dropdown2', None)
        if not dropdown_value :
           if dropdown_data :
               dropdown_value = dropdown_data.dropdown_value         
    
    if dropdown_value :
        if not dropdown_data:
            new_dropdown_data = DropdownData2(dropdown_value=dropdown_value)
            db.session.add(new_dropdown_data)
        else:
            dropdown_data.dropdown_value = dropdown_value
        db.session.commit()
    
    with current_app.app_context():
        exists = db.session.query(db.exists().where(InputData.align.isnot(None))).scalar()
        if exists:
            input_data = InputData.query.first()
            align = input_data.align
            global_constraint = input_data.global_constraint
            sakoe_chiba_radius = input_data.sakoe_chiba_radius
            itakura_max_slope = input_data.itakura_max_slope
            selected_vars = input_data.selected_vars

    step = read_counter()
    focus_json, _ = focus_plot2('init_data.db', step=step, global_constraint=global_constraint, 
                               sakoe_chiba_radius=sakoe_chiba_radius, itakura_max_slope=itakura_max_slope, 
                               align=align, dropdown_value=dropdown_value, selected_vars=selected_vars)
    return {'focus_json': focus_json}



#--------------- STORE DATA ------------------------#
@app.route('/store_data', methods=['POST'])
def store_data():
    data = request.get_json()
    align = data.get('align')
    global_constraint = data.get('global_constraint')
    sakoe_chiba_radius = data.get('sakoe_chiba_radius')
    itakura_max_slope = data.get('itakura_max_slope')
    dropdown_value = data.get('dropdown_value')
    selected_vars = str(data.get('selected_vars'))

    input_data = InputData.query.first()
    if input_data:
        # Update the existing row
        input_data.align = align
        input_data.global_constraint = global_constraint
        input_data.sakoe_chiba_radius = sakoe_chiba_radius
        input_data.itakura_max_slope = itakura_max_slope
        input_data.dropdown_value = dropdown_value
        input_data.selected_vars = selected_vars
    else:
        # Create a new row
        input_data = InputData(
            align=align,
            global_constraint=global_constraint,
            sakoe_chiba_radius=sakoe_chiba_radius,
            itakura_max_slope=itakura_max_slope,
            dropdown_value = dropdown_value,
            selected_vars = selected_vars
        )
        db.session.add(input_data)
    db.session.commit()
    return jsonify({'success': True})


#--------------- LOGIN ------------------ #
@app.route("/login", methods=['GET', 'POST'])
def login():
    app.empty_tables_flag = False  # Reset the flag when the user visits the home page
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else :
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template("login.html", title = 'Login', form = form)



@app.route("/register", methods = ['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, 
                    email=form.email.data,
                    password=hashed_password)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template("register.html", title = 'Register', form = form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))



def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/pics', picture_fn)
    
    output_size = (125, 125)
    i = Image.open(form_picture)
    i = ImageOps.exif_transpose(i)
    i.thumbnail(output_size)
    i.save(picture_path)
    
    return picture_fn


@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data :
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='pics/' + current_user.image_file)
    return render_template("account.html", title = 'Account', image_file=image_file, form=form)


@app.route("/documentation", methods=['GET', 'POST'])
def documentation():
    return render_template("documentation.html")



@app.route('/upload/last_df.csv')
def download_file():
    # Specify the path to the file
    file_path = '/home/baptiste/python/git_repository/web_app_copy/grasp/upload/last_df.csv'

    # Serve the file for download
    return send_file(file_path, as_attachment=True)


@app.before_request
def reset_file_uploaded():
    if not request.referrer:
        session['file_uploaded'] = False

