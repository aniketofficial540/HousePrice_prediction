from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())

    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/bedrooms')
def get_bedrooms():
    bedrooms = sorted(data['beds'].unique())
    return jsonify(bedrooms)

@app.route('/bathrooms')
def get_bathrooms():
    bathrooms = sorted(data['baths'].unique())
    return jsonify(bathrooms)

@app.route('/sizes')
def get_sizes():
    sizes = sorted(data['size'].unique())
    return jsonify(sizes)

@app.route('/zip_codes')
def get_zip_codes():
    zip_codes = sorted(data['zip_code'].unique())
    return jsonify(zip_codes)


@app.route('/predict', methods=['POST'])
def predict():
    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    size = request.form.get('size')
    zipcode = request.form.get('zip_code')

    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                              columns=['beds', 'baths', 'size', 'zip_code'])

    input_data['baths'] = pd.to_numeric(input_data['baths'], errors='coerce')
    input_data['baths'].fillna(data['baths'].mode()[0], inplace=True)

    input_data = input_data.astype({'beds': int, 'baths': float, 'size': float, 'zip_code': int})

    for column in input_data.columns:
        unknown_categories = set(input_data[column]) - set(data[column].unique())
        if unknown_categories:
            input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

    prediction = pipe.predict(input_data)[0]

    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5000
    )
