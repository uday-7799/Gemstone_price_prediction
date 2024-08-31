from src.DiamondPricePrediction.pipelines.prediction_pipeline import CustomData,PredictionPipeline

from flask import Flask,request,render_template,jsonify

# creating an object by passing current module(__name__)
app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template('form.html')
    else:
        try:
            carat = request.form.get('carat')
            depth = request.form.get('depth')
            table = request.form.get('table')
            x = request.form.get('x')
            y = request.form.get('y')
            z = request.form.get('z')

            # Validate numeric inputs
            if not all([carat, depth, table, x, y, z]):
                return render_template('form.html', error="Please provide all numeric inputs.")
            
            data = CustomData(
                carat=float(carat),
                depth=float(depth),
                table=float(table),
                x=float(x),
                y=float(y),
                z=float(z),
                cut=request.form.get('cut'),
                color=request.form.get('color'),
                clarity=request.form.get('clarity')
            )

            final_data = data.get_data_as_dataframe()

            # Create object for PredictionPipeline
            prediction_pipeline = PredictionPipeline()
            pred = prediction_pipeline.predict(final_data)

            result = round(pred[0], 2)

            return render_template('result.html', final_result=result)

        except ValueError:
            return render_template('form.html', error="Invalid input. Please enter valid numeric values.")
        except Exception as e:
            return render_template('form.html', error=f"An error occurred: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
