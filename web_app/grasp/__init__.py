from flask import Flask, session, request
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_dropzone import Dropzone

import os
import sqlite3

app = Flask(__name__)
app.config['SECRET_KEY'] = 'bzbil7vgj454gjh9076gd3r45Gre'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['UPLOAD_FOLDER'] = 'upload'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}
dropzone = Dropzone(app)
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'text/csv'
app.config['DROPZONE_MAX_FILE_SIZE'] = 5  # Max file size in megabytes
app.config['DROPZONE_MAX_FILES'] = 1  # Max number of files allowed
app.config['DROPZONE_UPLOAD_MULTIPLE'] = False
app.config['DROPZONE_PARALLEL_UPLOADS'] = 1


db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'



@app.before_request
def empty_tables():
    if 'loaded' not in session or request.referrer is None:
        print('LOADED IS NOT HERE')
        from grasp.models import InputData, CounterData, VerifData

        db.session.query(InputData).delete()
        db.session.query(CounterData).delete()
        db.session.query(VerifData).delete()
        db.session.query(CounterData).delete()
        db.session.commit()

        session['loaded'] = True

'''
@app.before_request
def before_request():
    if 'loaded' not in session or not session['loaded']:
        empty_tables()
'''

from grasp import routes

with app.app_context():
    db.create_all()