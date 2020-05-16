from flask import Flask, request
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/uploader', methods=['POST'])
def upload_file():
    f = request.files['file']
    f.save('./server_audio/data.wav')
    return 'file uploaded successfully'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
