import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from __future__ import division
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import pickle
import json

pd.set_option('display.max_columns', 100)

def data_prepare_train(json_file):
	df = pd.read_json(json_file, convert_dates=['approx_payout_date','event_created','event_end','event_published',                                               'event_start','sale_duration',                                               'sale_duration2','user_created',                                               'event_published'])
	df_new = df.loc[df.acct_type.isin(['premium','fraudster_event','fraudster','fraudster_att']), :]
	df_new['fraud'] = 0
	df_new.loc[df.acct_type.isin(['fraudster_event', 'fraudster', 'fraudster_att']), 'fraud'] = 1
	df_new['previous_payouts_total'] = df_new.previous_payouts.apply(len)
	# define features
	feature_cols = ['delivery_method', 'num_payouts', 'org_facebook', 'org_twitter', 'sale_duration',                 'previous_payouts_total']

	X = df_new[feature_cols]
	y = df_new.fraud

	df_new = pd.concat([X,y], axis = 1)
	df_new = df_new.dropna().copy()

	X_new = df_new[feature_cols]
	y_new = df_new.fraud

	X_train, X_test, y_train, y_test = train_test_split(X_new,y_new)

	rm = RandomForestClassifier(n_estimators=100)
	rm.fit(X_train, y_train)
	print(rm.score(X_test, y_test))
	print(roc_auc_score(y_test, rm.predict_proba(X_test)[:, 1]))
	print(confusion_matrix(y_test, rm.predict(X_test)))
	c = confusion_matrix(y_test, rm.predict(X_test))
	print(precision_score(y_test, rm.predict(X_test)))
	recall_score = c[1,1]/(c[1,1] + c[1,0])
	print(recall_score)
	filename = 'lihua_model.sav'
	pickle.dump(rm, open(filename, 'wb'))
	return rm

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


if __name__ == "__main__":
	# data_prepare_train('data.json')
	model_transform_predict('new_data_point.json')
	
