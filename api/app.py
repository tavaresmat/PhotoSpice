import matplotlib
matplotlib.use("Agg")
from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import os
import json
import base64
import io
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

from PySpice.Spice.Netlist import Circuit
from PySpice.Probe.Plot import plot
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Unit import *

from src.yolo_inference.netlist_generator import NetlistGenerator

# For the diode
IS = 4.352e-9
BOLTZMANN_CONSTANT = 1.380649e-23
FUNDAMENTAL_CHARG = 1.602e-19

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

raw_images_dir = "api/raw_images"

def eng_multiples(string):
            multiples = [
                ('k','e3'),
                ('m','e-3'),
                ('u','e-6'),
                ('n','e-9'),
                ('p','e-12')
            ]
            string = string.lower()
            for a,b in multiples:
                string = string.replace(a, b)
            return float(string)

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
    matplotlib.use("Agg")
    plt.switch_backend('agg')
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

    libraries_path = find_libraries('api\library')
    spice_library = SpiceLibrary(libraries_path)
    circuit.include(spice_library['1N4148'])

    if simulation_parameters["simulationType"] == "op":
        print("Starting simulation in operating point:")
        simulator = circuit.simulator()
        results = simulator.operating_point()
        return jsonify({
                    'msg': 'success',
                    'netlistResponse': [f"Node {str(node)}: {float(node):5.2f}" for node in results.nodes.values()]
        })

    elif simulation_parameters["simulationType"] == "transient":

        print("Starting simulation in transient mode:")
        plt.clf()
        simulator = circuit.simulator()
        simulation_parameters['temperature'] = 30 # FUTURE IMPLEMENTATIONS AAAAAAAAAAA
        print (
            eng_multiples(simulation_parameters["startTime"]),
            eng_multiples(simulation_parameters["stopTime"]), 
            eng_multiples(simulation_parameters["maximumStep"])
        )
        results = simulator.transient(
            start_time=eng_multiples(simulation_parameters["startTime"]),
            end_time=eng_multiples(simulation_parameters["stopTime"]), 
            step_time=eng_multiples(simulation_parameters["maximumStep"]),
            #ambient_temperature=simulation_parameters["temperature"] # future implementations 
        )
        if simulation_parameters["selected"].isnumeric():
            plt.plot(results.nodes[f"{simulation_parameters['selected']}".lower()].abscissa, 
                    results.nodes[f"{simulation_parameters['selected']}".lower()])
            plt.title(f"Node {simulation_parameters['selected']}")
            image_bytes = io.BytesIO()
            plt.savefig(image_bytes, format="png")
            image_bytes.seek(0)
            image_b64 = base64.b64encode(image_bytes.read())
            return jsonify({
                "netlistReponse": None,
                "voltageImg": image_b64.decode("utf-8"),
                "currentImg": ""
            })
        else:
            if simulation_parameters["selected"][0] in "lLVv":
                plt.plot(results.branches[f"{simulation_parameters['selected']}".lower()].abscissa, 
                        results.branches[f"{simulation_parameters['selected']}".lower()])
                plt.title(f"Current on {simulation_parameters['selected']}")
                image_bytes = io.BytesIO()
                plt.savefig(image_bytes, format="png")
                image_bytes.seek(0)
                image_b64 = base64.b64encode(image_bytes.read())
                return jsonify({
                    "netlistReponse": None,
                    "voltageImg": "",
                    "currentImg": image_b64.decode("utf-8")
                })
            elif simulation_parameters["selected"][0] in "Rr":
                searching = True
                for i, line in enumerate(new_netlist):
                    if searching == False:
                        break
                    for item in line:
                        if item == simulation_parameters["selected"]:
                            searching = False
                            node_plus = np.zeros(len(results.time)) if new_netlist[i][1] == "0" else results.nodes[f"{new_netlist[i][1]}"]
                            node_minus = np.zeros(len(results.time)) if new_netlist[i][2] == "0" else results.nodes[f"{new_netlist[i][2]}"]
                            current = (np.array(node_plus) - np.array(node_minus)) / eng_multiples(new_netlist[i][3])
                            break
                plt.plot(results.nodes["1"].abscissa, current)
                plt.title(f"Current on {simulation_parameters['selected']}")
                image_bytes = io.BytesIO()
                plt.savefig(image_bytes, format="png")
                image_bytes.seek(0)
                image_b64 = base64.b64encode(image_bytes.read())
                return jsonify({
                    "netlistReponse": None,
                    "voltageImg": "",
                    "currentImg": image_b64.decode("utf-8")
                })

            elif simulation_parameters["selected"][0] in "Cc":
                searching = True
                for i, line in enumerate(new_netlist):
                    if searching == False:
                        break
                    for item in line:
                        if item == simulation_parameters["selected"]:
                            searching = False
                            node_plus = np.zeros(len(results.time)) if new_netlist[i][1] == "0" else results.nodes[f"{new_netlist[i][1]}"]
                            node_minus = np.zeros(len(results.time)) if new_netlist[i][2] == "0" else results.nodes[f"{new_netlist[i][2]}"]
                            current = eng_multiples(new_netlist[i][3]) * np.ediff1d( np.array(node_plus) - np.array(node_minus)) \
                             * (1.0/eng_multiples(simulation_parameters["maximumStep"]))
                            break
                plt.plot(results.nodes["1"].abscissa[1:], current)
                plt.title(f"Current on {simulation_parameters['selected']}")
                image_bytes = io.BytesIO()
                plt.savefig(image_bytes, format="png")
                image_bytes.seek(0)
                image_b64 = base64.b64encode(image_bytes.read())
                return jsonify({
                    "netlistReponse": None,
                    "voltageImg": "",
                    "currentImg": image_b64.decode("utf-8")
                })
            
            elif simulation_parameters["selected"][1] in "D":
                searching = True
                for i, line in enumerate(new_netlist):
                    if searching == False:
                        break
                    for item in line:
                        if item == simulation_parameters["selected"]:
                            searching = False
                            node_plus = np.zeros(len(results.time)) if new_netlist[i][1] == "0" else results.nodes[f"{new_netlist[i][1]}"]
                            node_minus = np.zeros(len(results.time)) if new_netlist[i][2] == "0" else results.nodes[f"{new_netlist[i][2]}"]
                            vt = BOLTZMANN_CONSTANT * (simulation_parameters["temperature"] + 273)/(FUNDAMENTAL_CHARG)
                            current = IS * (np.e ** ( (np.array(node_plus) - np.array(node_minus))/vt) - 1)
                            break
                plt.plot(results.nodes["1"].abscissa, current)
                plt.title(f"Current on {simulation_parameters['selected']}")
                image_bytes = io.BytesIO()
                plt.savefig(image_bytes, format="png")
                image_bytes.seek(0)
                image_b64 = base64.b64encode(image_bytes.read())
                return jsonify({
                    "netlistReponse": None,
                    "voltageImg": "",
                    "currentImg": image_b64.decode("utf-8")
                })
                

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
