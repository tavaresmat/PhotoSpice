from flask import Flask, request, jsonify, session
from flask_cors import CORS
import cv2
import os
import base64

from src.yolo_inference.netlist_generator import NetlistGenerator


app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

raw_images_dir = "raw_images"
app.secret_key = b"laricalarica" # os.getenv("APP_SECRET_KEY").encode("ascii")
    

@app.route("/image", methods=["POST"])
def receive_image():
    image_id = len(os.listdir(raw_images_dir))
    session["image_id"] = image_id
    with open(f"{raw_images_dir}/{image_id}.png", 'wb') as file:
        file.write(request.data)
    
    response = jsonify({'msg': 'success', 'id': image_id })
    return response

@app.route("/netlist", methods=["GET"])
def send_netlist():
    if "image_id" not in session:
        return jsonify({"msg": "Please, send us an image first."})
    
    image_id = session["image_id"]
    image = cv2.imread(f"{raw_images_dir}/{image_id}.png")
    netlist_generator = NetlistGenerator()
    netlist = netlist_generator(image)
    debug_image = netlist_generator.debug_image

    sucess, data = cv2.imencode(".png", debug_image)
    # Improve error messages
    if sucess:
        data = base64.b64encode(data.tobytes())
    else:
        print("Ohhh shit, we don't have an image here")
        return {'msg': "sorry" }, 400

    session["image_id"] = image_id
    return jsonify({
                'msg': 'success',
                'debugImg': data.decode("utf-8"),
                'netlist': [line.split() for line in netlist.split("\n")]
           })

### A terminar
@app.route("/simulation", methods=["POST"])
def simulate():
    if "image_id" not in session:
        return jsonify({"msg": "Please, send us an image first."})
    
    new_netlist = request.form["new_netlist"]
    simulation_parameters = request.form["simulation_parameters"]
    
    

if __name__ == "__main__":
    app.run(debug=True)
