from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import os
import json
import base64

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

from src.yolo_inference.netlist_generator import NetlistGenerator


app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

raw_images_dir = "api/raw_images"

@app.route("/image", methods=["POST"])
def receive_image():
    image_id = len(os.listdir(raw_images_dir))
    with open(f"{raw_images_dir}/{image_id}.png", 'wb') as file:
        file.write(request.data)
    
    response = jsonify({'msg': 'success', 'id': image_id })
    return response

@app.route("/netlist/<id>", methods=["GET"])
def send_netlist(id):
    image_id = id
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

    return jsonify({
                'msg': 'success',
                'debugImg': data.decode("utf-8"),
                'netlist': [line.split() for line in netlist.split("\n")]
           })

### A terminar
@app.route("/simulation", methods=["POST"])
def simulate():
    simulation_data = json.loads(request.data)
    print(simulation_data)
    if "new_netlist" not in simulation_data.keys():
        return {'msg': "sorry, you need to send us back the netlist corrected!" }, 400
    new_netlist = simulation_data["new_netlist"]
    
    if "simulation_parameters" not in simulation_data.keys():
        return {'msg': "sorry, you need to send us the parameters for your simulation!" }, 400
    simulation_parameters = simulation_data["simulation_parameters"]

    raw_netlist = ""
    for sublist in new_netlist:
        line = ""
        for item in sublist:
            if "*" in item:
                values = item.split("*")
                item = f"SIN(0 {values[0]} {'1k' if len(values) <= 1 else values[1]})"
            line += " " + item
        line += "\n"
        raw_netlist += line

    circuit = Circuit("PhotoSpice Circuit")
    circuit.raw_spice = raw_netlist
    print(circuit)

    if simulation_parameters["simulationType"] == "op":
        print("Starting simulation in operating point:")
        simulator = circuit.simulator()
        results = simulator.operating_point()
        return jsonify({
                    'msg': 'success',
                    'netlistResponse': results.nodes.__repr__()
        })
    elif simulation_parameters["simulation_type"] == "transient":
        print("Starting simulation in transient mode:")
        simulator = circuit.simulator()
        results = simulator.transient()


if __name__ == "__main__":
    app.run(debug=True)
