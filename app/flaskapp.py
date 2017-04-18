# /usr/bin/python3
from flask import Flask
import json
import requests
import time
from datetime import datetime
import psycopg2
from sklearn.ensemble import RandomForestClassifier
import pickle


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

app = Flask(__name__)
PORT = 80
DATA = []
TIMESTAMP = []

rm = load_pickle('lihua_model.sav')
hostname = 'localhost'
username = 'postgres'
password = 'bcde1234'
database = 'dsifraud'
conn = psycopg2.connect( host=hostname, user=username, password=password, dbname=database )

def read_entry(example_path):
    '''
    Read single entry from http://galvanize-case-study-on-fraud.herokuapp.com/data_point
    '''
    with open(example_path) as data_file:
        d = json.load(data_file)
    df = pd.DataFrame()
    df_ = pd.DataFrame(dict([(k, pd.Series(d[k])) for k in d if (
        k != 'ticket_types') and (k != 'previous_payouts')]))
    df_['ticket_types'] = str(d['ticket_types'])
    df_['previous_payouts_total'] = len(d['previous_payouts'])
    df = df.append(df_)
    df.reset_index(drop=1, inplace=1)
    example = df
    return df

def data_prepare_test(json_file):

	df = read_entry(json_file)
	feature_cols = ['delivery_method', 'num_payouts', 'org_facebook', 'org_twitter', 'sale_duration','previous_payouts_total']
	X = df[feature_cols]
	return X.dropna()

def model_transform_predict(json_file):
	X = data_prepare_test(json_file)
	# print(X)
	filename = 'lihua_model.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	predicted = loaded_model.predict(X)
	print('Predicted Fraud Status is {}' .format(predicted[0]))
	return predicted[0]

def total_revenue(dicts):
    revenues = []
    for d in dicts:
        c = d['cost']
        sold = d['quantity_sold']
        revenue = c * float(sold)
        revenues.append(revenue)
    return sum(revenues)

def lifetime_value(revenue, num_days):
    if num_days != 0:
        periods = float(num_days)/365
        GC = revenue/periods
        CLV = GC * (1/1+0.04)
        return CLV
	    
def not_used():
	df['Revenue'] = df['ticket_types'].apply(lambda x: total_revenue(x))
	df['Lifetime Value'] = df.apply(lambda x: lifetime_value(x['Revenue'], x['user_age']), axis = 1)

	df_new_value = df_new.copy()
	df_new_value['ticket_types'] = df.ticket_types
	df_new_value['user_age'] = df.user_age

	df_new_value['Revenue'] = df_new_value['ticket_types'].apply(lambda x: total_revenue(x))
	df_new_value['Lifetime Value'] = df_new_value.apply(lambda x: lifetime_value(x['Revenue'], x['user_age']), axis = 1)
	df_new_value['predict'] = rm.predict(X_new)

	(df_new_value.fraud == df_new_value.predict).sum()


def get_datapoint():
    r = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point')
    DATA.append(json.dumps(r.json(), sort_keys=True, indent=4, separators=(',', ': ')))
    TIMESTAMP.append(time.time())


def store():
    cur = conn.cursor()
    result = []
    cur.execute( "select * from transactions;" )
    for text in cur.fetchall() :
        result.append(text)
    return result

@app.route('/helloworld/')
def helloworld():
    return "hello world!" 

@app.route('/')
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

@app.route('/check/')
def doquery():
    cur = conn.cursor()
    result = []
    cur.execute( "select * from transactions;" )
    for text in cur.fetchall() :
        result.append(text)
    return result



if __name__ == '__main__':
    get_datapoint()
    app.run()
