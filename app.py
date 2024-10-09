from flask import Flask, render_template, send_file
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/homework1')
def homework1():
    return render_template('homework1.html')

@app.route('/plot')
def plot():
    # Step 1: Generate synthetic data
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # Step 2: Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Step 3: Make predictions
    X_new = np.array([[0], [2]])
    y_predict = model.predict(X_new)

    # Step 4: Visualize the results
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X_new, y_predict, color='red', label='Linear Regression Line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Example')
    plt.legend()

    # Save plot to a BytesIO object and send it as a response
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
