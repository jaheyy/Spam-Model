from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('spamClassifierModel.pkl', 'rb'))
count_vectorizer = pickle.load(open('countVectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    print(request.form.values)
    message = count_vectorizer.transform(request.form.values()).toarray()
    prediction = model.predict(message)
    for i in message:
        print(i)

    output = prediction[0]
    prediction_text = ""

    if output == 0:
        prediction_text = "Not spam"
    elif output == 1:
        prediction_text = "Spam"

    return render_template('index.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)
