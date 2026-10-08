from flask import Flask, request
app = Flask(__name__)
@app.before_request
def before():
    try:
        print("Endpoint:", request.endpoint)
        print("Blueprint:", request.blueprint)
    except Exception as e:
        print("Error:", str(e))
@app.route('/')
def index():
    return 'Hello'

if __name__ == '__main__':
    with app.test_request_context('/not-found'):
        app.preprocess_request()
