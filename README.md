# deepspeech2_new

## How to Run
1. Create a VM instance (on Google Cloud Platform)
2. Clone the project
3. Download the data
4. Setup the enviroment (tensorflow v1.15)
5. Train the model

Some basic commands are:
```
// Connect to VM instance
gcloud compute instances start instance-name
gcloud compute ssh instance-name

// Activate the virtual environment 
source ./venv/bin/activate


python3 -W ignore ./train.py ./check_point/ ./json/for_newEngine/train_json.json ./json/for_newEngine/test_json.json ./log.csv

// Using screen for non-stop training
Create: screen 
Enter: screen -r
Exit: Ctrl + A D
```
