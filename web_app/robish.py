@app.route("/alignments", methods=['GET', 'POST'])
def alignments():

    min_count = 0
    max_count = 2
    align = 0
    inspector = inspect(db.engine)
    last_df = pd.DataFrame()

    # Instantiate the value of the counter
    if 'counter_data' in inspector.get_table_names():
        counter_data = CounterData.query.first()
        if hasattr(counter_data, 'counter'):
            counter = counter_data.counter
        else :
            counter = 0
    else :
        counter = 0
    counter = int(counter)

    form = UploadFileForm()
    file_uploaded = False
    if 'file_uploaded' in session:
        file_uploaded = session['file_uploaded']
        session.pop('file_uploaded', None)

    # Upload
    path_init, path_dynam, path_focus = [], [], []
    if form.validate_on_submit() or file_uploaded == True:
        if form.validate_on_submit():
            file = form.file.data
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            session['file_path'] = file_path
        else :
            file_path = session['file_path']

        # PLOT df initial
        df_init = pd.read_csv(file_path)
        max_count = len(df_init.columns)
        path_init = overall_plot(df_init, app.root_path, 'initial_plot')

        # PLOT df dynamique
        try :
            dynam_df
        except :
            dynam_df = df_init.copy()

        if 'input_data' in inspector.get_table_names():
            input_data = InputData.query.first()
            if hasattr(input_data, 'align'):
                align = input_data.align
        if (int(align) == 1) :
            prominence = float(input_data.input_prominence)
            window = int(input_data.input_dtw_window)
            dynam_df = transform(dynam_df, counter, prominence, window)      
            path_dynam = overall_plot(dynam_df, app.root_path, 'dynamic_plot')    
            
        else:
            print('didnt work')
            path_dynam = None
            path_focus = None

        file_uploaded = True
        session['file_uploaded'] = True
        
        # PLOT Focus
        focus_plot(dynam_df, counter, app.root_path)

        last_df = dynam_df

    csv_filepath = app.root_path + '/upload/last_df.csv'
    print(app.root_path)
    last_df.to_csv(csv_filepath, index=False)


    return render_template("alignments.html", form=form, file_uploaded=file_uploaded,
                           init_url=path_init, dynam_url=path_dynam,
                           csv_filepath=csv_filepath,
                           counter=counter, focus_url=path_focus, 
                           max_count=max_count, min_count=min_count, align=align)