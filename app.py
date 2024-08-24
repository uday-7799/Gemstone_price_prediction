from src.DiamondPricePrediction.pipelines.prediction_pipeline import CustomData,PredictionPipeline

from flask import Flask,request,render_template,jsonify

# creating an object by passing current module(__name__)
app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods = ["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template('form.html')
    
    else:
        data = CustomData(
            carat = float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get('cut'),
            color = request.form.get('color'),
            clarity = request.form.get('clarity')

        )

    final_data = data.get_data_as_dataframe()

    # creating object for PredictionPipeline
    prediction_pipeline = PredictionPipeline()

    pred = prediction_pipeline.predict(final_data)

    result = round(pred[0],2)

    return render_template('/result.html',final_result=result)

if __name__ == '__main__':
    app.run(debug=True)
