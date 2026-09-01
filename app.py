from flask import Flask, request, jsonify,render_template
from flask_cors import CORS

from model import analyze_project

app = Flask(__name__)

CORS(app)


@app.route("/")
def home():
    return render_template("sih3.html")
    

@app.route("/analyze", methods=["POST"])
def analyze():

    print("🔥 ANALYZE REQUEST RECEIVED")

    try:

        data = request.get_json()

        print("DATA RECEIVED:", data)

        result = analyze_project(data)

        print("AI RESULT:", result)

        return jsonify(result)

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    print("================================")
    print(" MPLADS AI SENTINEL")
    print("================================")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )