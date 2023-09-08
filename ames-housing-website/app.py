from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__, template_folder="templates")

# Load the CatBoost model
with open('best_model_catBoost.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the order of columns in X_train
column_order = ['lot_area', 'neighborhood', 'bldg_type', 'house_style', 'overall_qual',
                'overall_cond', 'year_built', 'year_remod/add', 'bsmt_unf_sf',
                'heating', 'central_air', '1st_flr_sf', '2nd_flr_sf', 'total_sqr_ft',
                'bedrooms', 'kitchen_qual', 'total_rooms', 'garage_type', 'car_garage',
                'garage_area', 'wood_deck_sf', 'open_porch_sf', 'enclosed_porch',
                'screen_porch', 'mo_sold', 'yr_sold', 'total_bathrooms',
                'remodeled_y/n', 'total_bsmtfin_sf', 'finbsmt/gr_liv_area',
                'avg_cs_index_value']



@app.route('/')
def index():
    overall_qual = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    house_style = ['Single Story', 'Two Story', 'Finished 1 1/2 Story',
                   'Split Foyer Style', 'Split Level Style', 'Unfinished 2 1/2 Story',
                   'Unfinished 1 1/2 Story', 'Finished 2 1/2 Story']
    remodel = ["Yes", "No"]
    neighborhood = ['NAmes', 'Gilbert', 'StoneBr', 'NWAmes', 'Somerst', 'BrDale', 'NPkVill',
                    'NridgHt', 'Blmngtn', 'NoRidge', 'SawyerW', 'Sawyer', 'Greens', 'BrkSide',
                    'OldTown', 'ClearCr', 'SWISU', 'Edwards', 'IDOTRR', 'CollgCr', 'Mitchel',
                    'MeadowV', 'Crawfor', 'Blueste', 'Timber', 'Veenker', 'GrnHill', 'Landmrk']
    central_air = ["Yes", "No"]
    default_lot_area = 0
    default_year_built = 2003
    default_total_rooms = 0
    default_bedrooms = 0
    default_bath = 0
    default_garage = 0

    user_input = {
        "overall_qual": overall_qual[0],
        "lot_area": default_lot_area,
        "house_style": house_style[0],
        "year_built": default_year_built,
        "remodel": remodel[0],
        "neighborhood": neighborhood[0],
        "central_air": central_air[0],
        "total_sqr_ft": 0,
        "total_rooms": default_total_rooms,
        "bedrooms": default_bedrooms,
        "bath": default_bath,
        "car_garage": default_garage
    }

    return render_template('index.html',
                           overall_qual=overall_qual,
                           lot_area=default_lot_area,
                           house_style=house_style,
                           year_built=default_year_built,
                           remodel=remodel,
                           neighborhood=neighborhood,
                           central_air=central_air,
                           default_total_rooms=default_total_rooms,
                           default_bedrooms=default_bedrooms,
                           default_bath=default_bath,
                           default_garage=default_garage,
                           user_input=user_input
                           )

@app.route('/predict', methods=['POST'])
def predict():
    # gather user input data
    user_input = {
        "overall_qual": request.form["overall_qual"],
        "lot_area": float(request.form["lot_area"]),
        "house_style": request.form["house_style"],
        "year_built": int(request.form["year_built"]),
        "remodel": request.form["remodel"],
        "neighborhood": request.form["neighborhood"],
        "central_air": request.form["central_air"],
        "total_sqr_ft": int(request.form["total_sqr_ft"]),
        "total_rooms": int(request.form["total_rooms"]),
        "bedrooms": int(request.form["bedrooms"]),
        "bath": float(request.form["bath"]),
        "car_garage": int(request.form["car_garage"])
    }

    # Preprocess the user input data
    user_input_df = pd.DataFrame([user_input], columns=column_order)

    # Fill missing values with mode from X_train
    X_train = pd.read_csv('data/training_data.csv')
    for column in user_input_df.columns:
        if column in X_train.columns:
            mode_value = X_train[column].mode()[0]
            user_input_df[column].fillna(mode_value, inplace=True)
        else:
            if user_input_df[column].dtype == 'object':
                user_input_df[column].fillna(mode_value, inplace=True)
            else:
                user_input_df[column].fillna(mode_value, inplace=True)

    # Perform model prediction
    prediction = model.predict(user_input_df)

    # Convert the prediction result to a meaningful output (e.g., JSON)
    result = {"prediction": prediction[0]}

    return render_template('index.html',
                           predicted_price=result["prediction"],
                           overall_qual=user_input["overall_qual"],
                           lot_area=user_input["lot_area"],
                           house_style=user_input["house_style"],
                           year_built=user_input["year_built"],
                           remodel=user_input["remodel"],
                           neighborhood=user_input["neighborhood"],
                           central_air=user_input["central_air"],
                           sqr_ft_house=user_input["total_sqr_ft"],
                           total_rooms=user_input['total_rooms'],
                           bedrooms=user_input["bedrooms"],
                           default_bath=user_input["bath"],
                           default_garage=user_input["car_garage"],
                           user_input=user_input
                           )

if __name__ == '__main__':
    app.run(debug=True)
