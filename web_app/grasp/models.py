from grasp import db, login_manager
from flask_login import UserMixin

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    __tablename__ = 'user_log'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(50), unique=True, nullable=False)
    image_file = db.Column(db.String(20), nullable = False, default="anonymous_user.jpg")
    password = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}', '{self.image_file}')"
    

class CounterData(db.Model):
    __tablename__ = 'counter'
    id = db.Column(db.Integer, primary_key=True)
    counter = db.Column(db.Integer)


class InputData(db.Model):
    __tablename__ = 'input_data'
    id = db.Column(db.Integer, primary_key=True)
    align = db.Column(db.String(100))
    global_constraint = db.Column(db.String(100))
    sakoe_chiba_radius = db.Column(db.String(10))
    itakura_max_slope = db.Column(db.String(10))
    dropdown_value = db.Column(db.String(100))


class VerifData(db.Model):
    __tablename__ = 'verif_features'
    id = db.Column(db.Integer, primary_key=True)
    global_constraint = db.Column(db.String(100))
    sakoe_chiba_radius = db.Column(db.Integer)
    itakura_max_slope = db.Column(db.Float)
    var1_dtw = db.Column(db.String)
    var2_dtw = db.Column(db.String)

