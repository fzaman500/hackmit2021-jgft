from flask import Flask, render_template, request
from gestureTracker import coordinates
import pandas

app = Flask(__name__)
print('Coordinates: ', coordinates)


@app.route('/')
def main(coordinates=coordinates):
    return render_template('index.html', coordinates=coordinates)

if __name__ == ' __main__':
    app.debug = True
    app.run()

