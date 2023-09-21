from flask import Flask, session, request
from flask_sqlalchemy import SQLAlchemy
from flask_dropzone import Dropzone
from flask_migrate import Migrate

import os
import sqlite3

# Initialize a Flask web application
app = Flask(__name__)

# Set a secret key for session security
app.config['SECRET_KEY'] = 'bzbil7vgj454gjh9076gd3r45Gre'

# Configure the application to use an SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

# Define the folder for file uploads
app.config['UPLOAD_FOLDER'] = 'upload'

# Specify allowed file extensions for uploads
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}

# Configure Dropzone settings for file uploads
dropzone = Dropzone(app)
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'text/csv'
app.config['DROPZONE_MAX_FILE_SIZE'] = 5  # Max file size in megabytes
app.config['DROPZONE_MAX_FILES'] = 1  # Max number of files allowed
app.config['DROPZONE_UPLOAD_MULTIPLE'] = False
app.config['DROPZONE_PARALLEL_UPLOADS'] = 1

# Initialize SQLAlchemy for database operations
db = SQLAlchemy(app)

# Initialize Flask-Migrate for database migrations
migrate = Migrate(app, db)


# Define a before-request function to reset data for each new session
@app.before_request
def empty_tables():
    if 'loaded' not in session or request.referrer is None or session['loaded'] == None:
        # Import database models
        from grasp.models import InputData, CounterData, VerifData, HistoricData, DropdownData1, DropdownData2

        # Delete data from various tables in the database
        db.session.query(InputData).delete()
        db.session.query(CounterData).delete()
        db.session.query(VerifData).delete()
        db.session.query(HistoricData).delete()
        db.session.query(DropdownData1).delete()
        db.session.query(DropdownData2).delete()
        
        # Commit changes to the database
        db.session.commit()

        # Mark the session as loaded
        session['loaded'] = True

        # Remove a temporary SQLite database file if it exists
        conn = sqlite3.connect('temporary_data.db')
        app_path = os.path.dirname(app.root_path)
        if os.path.exists(os.path.join(app_path, 'temporary_data.db')):
            os.remove(os.path.join(app_path, 'temporary_data.db'))
        conn.close()

# Import routes from the grasp module
from grasp import routes

# Create the database tables
with app.app_context():
    db.create_all()
