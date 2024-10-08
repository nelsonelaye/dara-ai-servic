from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from models.blenderbot_model import chat
# import train

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    data = {
        "message":"Hello Dara"
    }
    return jsonify(data), 200

@app.post("/chat")
def start_chat():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            # Return 400 error if "text" is not provided in the request
            return jsonify({"error": "Bad request", "message": "'text' is required"}), 400
        
        query = data.get("text")
        chat_response=""
        if query:
            chat_response = chat(query)
        else:
            chat_response =  chat("")
        
        data= {
            "message": chat_response
        }
        return jsonify(data), 200
    except Exception  as e:
        return jsonify({"error": "An internal error occurred", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)