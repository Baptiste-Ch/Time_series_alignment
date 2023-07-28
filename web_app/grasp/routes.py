import os
import sys
import secrets
from PIL import Image, ImageOps
from flask import render_template, url_for, flash, redirect, request, session, jsonify, send_file, current_app
from grasp import app, db, bcrypt
from grasp.models import User, InputData, CounterData
from grasp.forms import RegistrationForm, LoginForm, UpdateAccountForm
from flask_login import login_user, current_user, logout_user, login_required


import pandas as pd
import numpy as np
import time
import json

import sqlite3


sys.path.append(app.root_path)

from functions import overall_plot, focus_plot, copy_tables, read_counter, increment_counter, reorder_tables



#--------------- HOME ------------------ #
@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')


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

        print()

        return jsonify({"message": "Tables transferred successfully!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


#------------NEXT DATA---------------------#
@app.route('/next_data', methods=['GET', 'POST'])
def next_data():
    copy_tables('temporary_data.db', 'modified_data.db')
    reorder_tables('init_data.db', 'modified_data.db')
    increment_counter()

    step = read_counter()
    if not step:
        step = 0
    modified_json, modified_dropdown = overall_plot('modified_data.db')
    focus_json, focus_dropdown = focus_plot('init_data.db', step)

    response_data = {
        'modified_json': modified_json,
        'focus_json': focus_json,
        'modified_dropdown': modified_dropdown,
        'focus_dropdown': focus_dropdown
    }

    print('NEXT DATA')
    return jsonify(response_data)


#-------------- UPLOAD ---------------#
@app.route('/upload', methods=['POST'])
def upload():
    print('UPLOAD')
    file = request.files['file']
    file_path = os.path.join(app.root_path, 'upload', file.filename)
    file.save(file_path)
    session['uploaded_file_path'] = file_path
    
    df = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')

    # Create a SQLite database and connect to it
    for i in ['init_data.db', 'modified_data.db'] :
        path_to_delete = os.path.join(app.root_path, i)
        if os.path.exists(path_to_delete):
            os.remove(path_to_delete)
    conn_1 = sqlite3.connect('init_data.db')
    conn_2 = sqlite3.connect('modified_data.db')

    # Loop over the sheets in the Excel file
    for i in [conn_1, conn_2] :
        for sheet_name, sheet_data in df.items():
            # Store the data in a table in the SQLite database
            sheet_data.to_sql(sheet_name, i, if_exists='replace')
        i.close()

    return redirect(url_for('alignments', file_path=file_path))


#--------------- ALIGNMENTS ------------------ #
@app.route("/alignments", methods=['GET', 'POST'])
def alignments():    
    print('ALIGNMENTS')
    init_json, init_dropdown = overall_plot('init_data.db')
    modified_json, modified_dropdown = overall_plot('modified_data.db')

    step = read_counter()
    if not step:
        step = 0
    focus_json, focus_dropdown = focus_plot('init_data.db', step)
    return render_template('alignments.html', init_json=init_json, init_dropdown=init_dropdown,
                           modified_json=modified_json, modified_dropdown=modified_dropdown,
                           focus_json=focus_json, focus_dropdown=focus_dropdown)


#--------------- ALIGNMENTS - FIGURE_INIT -----------#
@app.route("/alignments/figure_init", methods=['POST'])
def figure_init():
    init_json, _ = overall_plot('init_data.db')
    return jsonify(init_json=init_json)


#--------------- ALIGNMENTS - FIGURE_MODIFIED -----------#
@app.route("/alignments/figure_modified", methods=['POST'])
def figure_modified():
    modified_json, _ = overall_plot('modified_data.db')
    return jsonify(modified_json=modified_json)


#--------------- ALIGNMENTS - FIGURE_FOCUS -----------#
@app.route("/alignments/figure_focus", methods=['POST'])
def figure_focus():
    print('FIGURE FOCUS')
    align=None
    global_constraint=None
    sakoe_chiba_radius=None
    itakura_max_slope=None
    dropdown_value=None
    with current_app.app_context():
        exists = db.session.query(db.exists().where(InputData.align.isnot(None))).scalar()
        if exists:
            input_data = InputData.query.first()
            align = input_data.align
            global_constraint = input_data.global_constraint
            sakoe_chiba_radius = input_data.sakoe_chiba_radius
            itakura_max_slope = input_data.itakura_max_slope
            dropdown_value = input_data.dropdown_value

    step = read_counter()
    focus_json, _ = focus_plot('init_data.db', step=step, global_constraint=global_constraint, sakoe_chiba_radius=sakoe_chiba_radius, 
                               itakura_max_slope=itakura_max_slope, align=align, dropdown_value=dropdown_value)
    return {'init_json': focus_json}



@app.route('/store_data', methods=['POST'])
def store_data():
    data = request.get_json()
    align = data.get('align')
    global_constraint = data.get('global_constraint')
    sakoe_chiba_radius = data.get('sakoe_chiba_radius')
    itakura_max_slope = data.get('itakura_max_slope')
    dropdown_value = data.get('dropdown_value')
    input_data = InputData.query.first()
    if input_data:
        # Update the existing row
        input_data.align = align
        input_data.global_constraint = global_constraint
        input_data.sakoe_chiba_radius = sakoe_chiba_radius
        input_data.itakura_max_slope = itakura_max_slope
        input_data.dropdown_value = dropdown_value
    else:
        # Create a new row
        input_data = InputData(
            align=align,
            global_constraint=global_constraint,
            sakoe_chiba_radius=sakoe_chiba_radius,
            itakura_max_slope=itakura_max_slope,
            dropdown_value = dropdown_value
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

