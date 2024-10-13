from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/coral')
def coral():
    return render_template('coral.html')

if __name__ == '__main__':
    app.run(debug=True)

