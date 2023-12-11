from flask import Flask, make_response, jsonify, request
from flask_cors import CORS, cross_origin
from Photovoltaik_predictor import PhotovoltaikPredictor
from Wind_Off_Shore_predictor import WindOffShorePredictor
from Wind_On_Shore_predictor import WindOnShorePredictor
from Power_Usage_predictor import PowerPredictor
from Price_predictor import PricePredictor


photovoltaik_predictor = PhotovoltaikPredictor()
wind_on_shore_predictor = WindOnShorePredictor()
wind_off_shore_predictor = WindOffShorePredictor()
power_predictor = PowerPredictor()
price_predicotr = PricePredictor()

app = Flask('app')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

 
@app.route('/predict_photovoltaik', methods=['GET'])
@cross_origin()
def predict_photovoltaik():
    
    forecast_days = request.args.get('forecast_days', type=int)
    latitude = request.args.get('forecast_days', type=float)
    longitude = request.args.get('forecast_days', type=float)
    pWk = request.args.get('forecast_days', type=int)

    result = photovoltaik_predictor.get_Prediction(forecast_days, latitude, longitude, pWk)
    
    return make_response(
            result.to_json(orient="split"),
            200
    )

@app.route('/predict_on_shore', methods=['GET'])
@cross_origin()
def predict_on_shore():
    
    forecast_days = request.args.get('forecast_days', type=int)
    
    result = wind_on_shore_predictor.get_Prediction(forecast_days)
    
    return make_response(
            result.to_json(orient="split"),
            200
    )
    
@app.route('/predict_off_shore', methods=['GET'])
@cross_origin()
def predict_off_shore():
    
    forecast_days = request.args.get('forecast_days', type=int)
    
    result = wind_off_shore_predictor.get_Prediction(forecast_days)
    
    return make_response(
            result.to_json(orient="split"),
            200
    )
    
@app.route('/predict_power_usage', methods=['GET'])
@cross_origin()
def predict_power_usage():

    result = power_predictor.get_Prediction()
    
    return make_response(
            result.to_json(orient="split"),
            200
    )
    
@app.route('/predict_price', methods=['GET'])
@cross_origin()
def predict_price():

    result = price_predicotr.get_Prediction()
    
    return make_response(
            result.to_json(orient="split"),
            200
    )
    


if __name__ == "__main__":

   app.run(host="0.0.0.0", debug=True)