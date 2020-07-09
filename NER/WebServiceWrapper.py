import DataExtraction
from flask import Flask, jsonify, request


app = Flask(__name__)

@app.route("/")
def default():
    return "Hello world>>!!"

@app.route('/structure/<unstructureddata>',methods=['GET'])
def structure(unstructureddata):
    structureddata = DataExtraction.returnStructuredData(unstructureddata)
    return jsonify(structureddata)

if __name__ == "__main__":
    app.run()
