# Build Docker Image

docker buildx build . -t leonidasmeurer/frederick_api:1.0
docker run -p 127.0.0.1:5002:5002 leonidasmeurer/frederick_api:1.0
docker push leonidasmeurer/frederick_api:1.0

Go to: 127.0.0.1:8080


# Pull Docker Image

docker pull leonidasmeurer/frederick_api:1.1
docker run -p 127.0.0.1:5002:5002 leonidasmeurer/frederick_api:1.1

Go to: 127.0.0.1:5002

# API
API:
predict_photovoltaik:

adress:predict_photovoltaik / predict_on_shore / predict_off_shore

args: forecast_days: int

returns: json mit timestamp und MwH:
{"0":
"1701820800000":36.935131073,"1701824400000":25.624540329,"1701828000000":29.3109378815, ... }

# Bsp:
http://127.0.0.1:5002/predict_photovoltaik?forecast_days=3
http://127.0.0.1:5002/predict_on_shore?forecast_days=2
http://127.0.0.1:5002/predict_off_shore?forecast_days=7



