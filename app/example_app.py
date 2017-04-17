from flask import Flask
import json
import requests
import time
from datetime import datetime

app = Flask(__name__)
PORT = 5353
DATA = []
TIMESTAMP = []

def get_datapoint():
    r = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point')
    DATA.append(json.dumps(r.json(), sort_keys=True, indent=4, separators=(',', ': ')))
    TIMESTAMP.append(time.time())


@app.route('/check')
def check():
    line1 = "Number of data points: {0}".format(len(DATA))
    if DATA and TIMESTAMP:
        dt = datetime.fromtimestamp(TIMESTAMP[-1])
        data_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        line2 = "Latest datapoint received at: {0}".format(data_time)
        line3 = DATA[-1]
        output = "{0}\n\n{1}\n\n{2}".format(line1, line2, line3)
    else:
        output = line1
    return output, 200, {'Content-Type': 'text/css; charset=utf-8'}


if __name__ == '__main__':
    get_datapoint()
    app.run(host='0.0.0.0', port=PORT, debug=True)
