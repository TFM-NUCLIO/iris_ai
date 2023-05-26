from flask import Flask, render_template, request, url_for
import os

app = Flask(
  __name__,
  template_folder='templates',
  static_folder='static'
)

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/calculate',methods = ['POST', 'GET'])
def calculate():
    if request.method == 'POST':
        value1 = request.form['value1']
        value2 = request.form['value2']
    else:
        value1 = request.args.get('value1')
        value2 = request.args.get('value2')
    return "calculate " + str(value1) + str(value2)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port, threaded=True)