# Time_series_alignment

Pour faire tourner l'application :
  - téléchargrer le dossier web_app ;
  - ouvrir un terminal de commande dans ce dossier (clique droit : '_Ouvrir un terminal ici_' OU commande _cd_) ;
  - taper dans le terminal : python run.py ;
  - Ouvrir votre navigateur web (si problème prioriser FireFox) et aller sur : http://127.0.0.1:5000/alignments ;

Le format de la donnée d'entrée doit correspondre à celui présenté dans _initial_dataset_tests.xlsx_. C'est à dire :
  - Une feuille par carotte ;
  - Variable X en première colonne ;
  - Toutes les variables au même nom ;

Packaging nécessaire :
  - flask
  - flask_sqlalchemy
  - flask_session
  - flask_bcrypt
  - flask_login
  - flask_dropzone
  - sqlite3
  - json
  - secrets
  - PIL

  - numpy
  - pandas
  - plotly
  - tslearn
  - sklearn
  - scipy
