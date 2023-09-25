from grasp import db


class CounterData(db.Model):
    """
    Base comptant à quelle séquence nous sommes dans l'ensemble des tables.
    Défini aussi le nombre total de séquences.
    """
    __tablename__ = 'counter'
    id = db.Column(db.Integer, primary_key=True)
    counter = db.Column(db.Integer)
    max_counter = db.Column(db.Integer)


class InputData(db.Model):
    """
    InputData contient les hyperparamètres défini par l'utilisateur.
    Les données de la table sont mis à jour lorsque l'on clique sur 'Run Function'.

    Vars :
        - id : Index de la table
        - align : Valeur pour indiquer à focus_plot et focus_plot2 d'effectuer l'alignement ou non
        - global_constraint : Paramètre global_constraint.
        - sakoe_chiba_radius : Paramètre sakoe_chiba_radius
        - itakura_max_slope : Paramètre itakura_max_slope
        - dropdown_value : Variable à afficher, provient du boutton : dropdwon_value1 ou dropdwon_value2
        - selected_vars : Liste des variables avec lequel on aligne (provient de checkbox)
    """
    __tablename__ = 'input_data'
    id = db.Column(db.Integer, primary_key=True)
    align = db.Column(db.String(100))
    global_constraint = db.Column(db.String(100))
    sakoe_chiba_radius = db.Column(db.String(10))
    itakura_max_slope = db.Column(db.String(10))
    dropdown_value = db.Column(db.String(100))
    selected_vars = db.Column(db.String(1000))


class VerifData(db.Model):
    """
    VerifData contient les hyperparamètres défini par l'utilisateur AU PRECEDENT ALIGNEMENT.
    Les données de la table sont mis à jour lorsque l'on clique sur 'Run Function'.

    Vars :
        - id : Index de la table
        - align : Valeur pour indiquer à focus_plot et focus_plot2 d'effectuer l'alignement ou non
        - global_constraint : Paramètre global_constraint.
        - sakoe_chiba_radius : Paramètre sakoe_chiba_radius
        - itakura_max_slope : Paramètre itakura_max_slope
        - dropdown_value : Variable à afficher, provient du boutton : dropdwon_value1 ou dropdwon_value2
        - selected_vars : Liste des variables avec lequel on aligne (provient de checkbox)
    """
    __tablename__ = 'verif_features'
    id = db.Column(db.Integer, primary_key=True)
    global_constraint = db.Column(db.String(100))
    sakoe_chiba_radius = db.Column(db.Integer)
    itakura_max_slope = db.Column(db.Float)
    selected_vars = db.Column(db.String(1000))
    var1_dtw = db.Column(db.String)
    var2_dtw = db.Column(db.String)


class DropdownData1(db.Model):
    """
    Variable d'intérêt pour les visualisation globales (init.db & modified_.db)
    Récupère la valeur lorsque l'on modifie dropdown-container1
    """
    __tablename__ = 'dropdown_data1'
    id = db.Column(db.Integer, primary_key=True)
    dropdown_value = db.Column(db.String(100))


class DropdownData2(db.Model):
    """
    Variable d'intérêt pour figure_focus
    Récupère la valeur lorsque l'on modifie dropdown-container1
    """
    __tablename__ = 'dropdown_data2'
    id = db.Column(db.Integer, primary_key=True)
    dropdown_value = db.Column(db.String(100))


class HistoricData(db.Model):
    """
    Historique des fonctions d'alignement lors de chaque étapes. 
    Nécessaire pour 'Option1' pour effectuer les alignements de toutes les séquences

    Vars :
        - id : Index de la table
        - lgth_last_elem : taille de la dernière séquence (nécessaire pour interpoler 
          les données t sur les données t-1 pour ensuite faire l'alignement)
        - var2_dtw : function d'alignement du df2 sur le df1
    """
    __tablename__ = 'historic'
    id = db.Column(db.Integer, primary_key=True)
    lgth_last_elem = db.Column(db.Integer)
    var2_dtw = db.Column(db.String)   
    min_depth = db.Column(db.String) 
    max_depth = db.Column(db.String) 

