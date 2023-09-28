# Time_series_alignment

Pour faire tourner l'application :
  - téléchargrer le dossier web_app ;
  - ouvrir un terminal de commande dans ce dossier (clique droit : '_Ouvrir un terminal ici_' OU commande _cd_) ;
  - taper dans le terminal : python run.py ;
  - Ouvrir votre navigateur web (si problème prioriser FireFox) et aller sur : http://127.0.0.1:5000/displays ;

Le format de la donnée d'entrée doit correspondre à celui présenté dans _initial_dataset_tests.xlsx_. C'est à dire :
  - Une feuille par carotte ;
  - Variable X en première colonne ;
  - Toutes les variables au même nom ;

Version python : 3.9.13

Packaging nécessaire :
  - flask : 2.2.3
  - flask_sqlalchemy : 3.0.3
  - flask_session : 0.5.0
  - flask_bcrypt : 1.0.1
  - flask_login : 0.6.2
  - flask_dropzone : 1.6.0
  - sqlite3 : 3.39.2
  - json : 
  - secrets
  - PIL

  - numpy
  - pandas
  - plotly
  - tslearn
  - sklearn
  - scipy


L'outil est encore en développement
