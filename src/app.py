import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import subprocess

app = Flask(__name__)
CORS(app)

# UPLOAD_FOLDER = "./uploads"
UPLOAD_FOLDER = r"C:\Users\tbezdan\Desktop\crAIRsis data"
PROJECT_FOLDER = ""
CSV_FOLDER = "original_datasets"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Retrieve the project_name parameter
    project_name = request.form.get(
        "project_name"
    )  # Get project_name from the form data

    # Validate that project_name is provided
    if not project_name:
        return jsonify({"error": "Project name is required"}), 400

    if file and file.filename.endswith(".csv"):
        global PROJECT_FOLDER
        PROJECT_FOLDER = os.path.join(UPLOAD_FOLDER, project_name)
        DATA_FOLDER = os.path.join(PROJECT_FOLDER, CSV_FOLDER)
        os.makedirs(DATA_FOLDER, exist_ok=True)

        file_path = os.path.join(DATA_FOLDER, file.filename)
        file.save(file_path)

        # Citanje CSV fajla i pribavljanje kolona
        df = pd.read_csv(file_path)
        columns = df.columns.tolist()

        response_data = {"filename": file.filename, "columns": columns}
        print(f"Response: {response_data}")  # Debug print

        # Save JSON to file
        # json_filename = os.path.splitext(file.filename)[0] + '.json'
        # json_file_path = os.path.join(UPLOAD_FOLDER, json_filename)
        # with open(json_file_path, 'w') as json_file:
        #    json.dump(response_data, json_file, indent=4)

        # Vrati JSON
        return (
            jsonify(
                {
                    "message": "First step OK",
                    "columns": columns,
                    "project folder": PROJECT_FOLDER,
                    "distribution": 12,
                }
            ),
            200,
        )

    return jsonify({"error": "File type not allowed"}), 400


@app.route("/configure", methods=["POST"])
def configure():
    config_data = request.json
    # Snimi konfiguraciju u user_config.json
    data_file_path = os.path.join(PROJECT_FOLDER, "user_config.json")
    with open(data_file_path, "w") as data_file:
        json.dump(config_data, data_file, indent=4)

    print(f"Saved to {data_file_path}")  # Debug print
    config_data["project_folder"] = PROJECT_FOLDER

    # Pozovi main.py u produkciji, u razvoju process_data.py i prosledi putanju za user_config.json
    process_script = "process_data.py"
    try:
        # Sinhrono, cekamo odgovor
        # result = subprocess.run(
        #     # ["python", process_script, data_file_path],
        #     ["python", process_script, data_file_path, PROJECT_FOLDER],
        #     capture_output=True,
        #     text=True,
        # )
        subprocess.Popen(["python", process_script, data_file_path, PROJECT_FOLDER])
        # if result.returncode != 0:
        #     return (
        #         jsonify(
        #             {"error": "Error running process_data.py", "details": result.stderr}
        #         ),
        #         500,
        #     )

        # Asinhrono
        # subprocess.Popen(['python3', process_script, data_file_path])

        # Vrati klijentu poruku daje sve proteklo OK
        return (
            jsonify(
                {
                    "message": "Configuration saved to user_config.json and processed",
                    "config": config_data,
                    "process_output": "Success",
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
