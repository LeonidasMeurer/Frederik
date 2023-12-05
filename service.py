from flask import Flask, make_response, jsonify, request
#from flask_cors import CORS, cross_origin
from Photovoltaik_Test_Model import PhotovoltaikPredictor
from Wind_Off_Shore_Test_Model import WindOffShorePredictor
from Wind_On_Shore_Test_Model import WindOnShorePredictor


photovoltaik_predictor = PhotovoltaikPredictor()
wind_on_shore_predictor = WindOnShorePredictor()
wind_off_shore_predictor = WindOffShorePredictor()

app = Flask('app')
#cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

 
@app.route('/predict_photovoltaik', methods=['GET'])
#@cross_origin()
def predict_photovoltaik():
    
    forecast_days = request.args.get('forecast_days', type=int)

    result = photovoltaik_predictor.get_Prediction(forecast_days)
    
    return make_response(
            result,
            200
    )

@app.route('/predict_on_shore', methods=['GET'])
#@cross_origin()
def predict_on_shore():
    
    forecast_days = request.args.get('forecast_days', type=int)
    print(forecast_days)

    result = wind_on_shore_predictor.get_Prediction(forecast_days)
    
    return make_response(
            result,
            200
    )
    
@app.route('/predict_off_shore', methods=['POST'])
#@cross_origin()
def predict_off_shore():
    
    forecast_days = request.args.get('forecast_days', type=int)
    print(forecast_days)

    result = wind_off_shore_predictor.get_Prediction(forecast_days)
    
    return make_response(
            result,
            200
    )
    


if __name__ == "__main__":

   app.run(host="0.0.0.0", debug=True)