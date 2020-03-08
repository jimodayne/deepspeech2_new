# deepspeech2_new

## How to Run
1. Create a VM instance (on Google Cloud Platform)
2. Clone the project
3. Download the data
4. Run some commnad below


## Useful commnad line
1. gcloud compute instances start instance-3
2. gcloud compute ssh instance-3 
3. source ./venv/bin/activate
4. python3 -W ignore ./train.py ./check_point/ ./json/for_newEngine/train_json.json ./json/for_newEngine/test_json.json ./log.csv
5. screen -r
6. exit: Ctrl + A D
