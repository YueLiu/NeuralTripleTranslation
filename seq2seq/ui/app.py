from flask import Flask, request, jsonify
from flask_cors import CORS
import get_model_api  # see part 1.
from flask import render_template
import json
app = Flask(__name__)
CORS(app) # needed for cross-domain requests, allow everything by default


# default route
@app.route('/')
def index():
    return ""

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404

@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

# API route
@app.route('/api', methods = ['POST'])
def api():
    input_data = request.json
    # app.logger.info("api_input: " + str(input_data))
    output_data = get_model_api.s2s_end2end(input_data)
    # app.logger.info("api_output: " + str(output_data))
    # response = jsonify(output_data)
    return render_template('table.html', response=output_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)