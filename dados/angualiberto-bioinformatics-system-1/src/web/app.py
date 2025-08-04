from flask import Flask, render_template, request, jsonify
from src.core.system import AnguaLibertoSystem

app = Flask(__name__)
system = AnguaLibertoSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    sequence = request.form.get('sequence')
    analysis_type = request.form.get('analysis_type', 'Geral')
    results = system.generate_complete_report(sequence, analysis_type)
    return jsonify(results)

@app.route('/report')
def report():
    return render_template('report.html')

if __name__ == '__main__':
    app.run(debug=True)