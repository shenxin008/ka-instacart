#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

def model():
    np.set_printoptions(suppress=True)
    # x = data_temp_num[['order_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']]
    # y = data_temp_num['predict']

# def model(data_all, x_list, y_list):
    # x = data_all[[x_list]]
    # y = data_all[y_list]
    # print x
    # print y

    # x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    lr = Pipeline([('sc', StandardScaler()),
                   ('poly', PolynomialFeatures(degree=1)),
                   ('clf', LogisticRegression())])
    lr.fit(x_train, y_train.ravel())
    y_hat = lr.predict(x_train)
    # y_hat_prob = lr.predict_proba(x_train)
    # print 'y_hat = \n', y_hat
    # print 'y_hat_prob = \n', y_hat_prob
    train_acc_rate = 100*np.mean(y_hat == y_train.ravel())
    print u'训练准确度：%.2f%%' % (train_acc_rate)
    train_acc_rate_list.append(train_acc_rate)

    test_y_hat = lr.predict(x_test)
    # test_y_hat_prob = lr.predict_proba(x_test)
    # print 'y_hat = \n', y_hat
    # print 'y_hat_prob = \n', y_hat_prob
    test_acc_rate = 100*np.mean(test_y_hat == y_test.ravel())
    print u'测试准确度：%.2f%%' % (test_acc_rate)
    test_acc_rate_list.append(test_acc_rate)

if __name__ == "__main__":
    # data_tr = pd.read_csv('order_products__train.csv')
    # print len(data_tr), data_tr['order_id'].drop_duplicates().describe()

    order_all = pd.read_csv('orders.csv')
    order_tr = order_all[order_all['eval_set'] == 'train']
    order_tr = order_tr[order_tr['user_id'] % 10 == 0]
    print len(order_tr)

    # order_te = order_all[order_all['eval_set'] == 'test']
    # print len(order_tr), len(order_te)

    order_tr_pri = order_all[order_all['user_id'].isin(order_tr['user_id'])]
    print len(order_tr_pri)
    # data_tr_pri = pd.DataFrame()
    data_tr_pri_list = []
    i = 0
    data_all = pd.read_csv('order_products__prior.csv', chunksize=1000000)
    for chunk in data_all:
        data_tr_pri_list.append(pd.merge(chunk, order_tr_pri, how='inner', on='order_id'))
        data_tr_pri = pd.concat(data_tr_pri_list, ignore_index=True)
    print len(data_tr_pri)

    # get uer level feature
    data_tr_pri['count_ord_user'] = data_tr_pri['order_id']
    data_tr_pri['count_pro_user'] = data_tr_pri['product_id']
    data_tr_pri['mean_order_dow_user'] = data_tr_pri['order_dow']
    data_tr_pri['mean_order_hour_of_day_user'] = data_tr_pri['order_hour_of_day']
    data_tr_pri['mean_days_since_prior_order_user'] = data_tr_pri['days_since_prior_order']
    data_tr_num = data_tr_pri.groupby('user_id').agg({'count_ord_user': pd.Series.nunique,
                                                      'count_pro_user': pd.Series.nunique,
                                                      'mean_order_dow_user': np.mean,
                                                      'mean_order_hour_of_day_user': np.mean,
                                                      'mean_days_since_prior_order_user': np.mean})
    data_tr_num['user_id'] = data_tr_num.index
    print len(data_tr_num), data_tr_num.head()

    data_tr_pro_list = data_tr_pri.product_id.unique()
    print len(data_tr_pro_list)

    order_tr2 = order_all[order_all['eval_set'] == 'train']
    data_tr = pd.read_csv('order_products__train.csv')
    data_tr = pd.merge(data_tr, order_tr2, how='left', on='order_id')
    data_tr['predict'] = 1
    data_tr = data_tr.loc[:, ['user_id', 'product_id', 'predict']]
    # print data_tr.head()
    train_acc_rate_list = []
    test_acc_rate_list = []
    i = 0
    for product_id in data_tr_pro_list:
        data_temp = data_tr_pri[data_tr_pri['product_id'] == product_id]
        data_temp['count'] = data_temp['order_id']
        data_temp['max_order_number'] = data_temp['order_number']
        data_temp['mean_order_dow'] = data_temp['order_dow']
        data_temp['min_order_dow'] = data_temp['order_dow']
        data_temp['max_order_dow'] = data_temp['order_dow']
        data_temp['mean_order_hour_of_day'] = data_temp['order_hour_of_day']
        data_temp['min_order_hour_of_day'] = data_temp['order_hour_of_day']
        data_temp['max_order_hour_of_day'] = data_temp['order_hour_of_day']
        data_temp['mean_days_since_prior_order'] = data_temp['days_since_prior_order']
        data_temp['min_days_since_prior_order'] = data_temp['days_since_prior_order']
        data_temp['max_days_since_prior_order'] = data_temp['days_since_prior_order']
        data_temp_num = data_temp.groupby('user_id').agg({'count': pd.Series.nunique,
                                                          'max_order_number': np.max,
                                                          'mean_order_dow': np.mean,
                                                          'min_order_dow': np.min,
                                                          'max_order_dow': np.max,
                                                          'mean_order_hour_of_day': np.mean,
                                                          'min_order_hour_of_day': np.min,
                                                          'max_order_hour_of_day': np.max,
                                                          'mean_days_since_prior_order': np.mean,
                                                          'min_days_since_prior_order': np.min,
                                                          'max_days_since_prior_order': np.max,})
        data_temp_num['user_id'] = data_temp_num.index
        data_temp_num = pd.merge(data_temp_num, data_tr_num, how='left', on='user_id')
        # print data_temp_num.head()
        # break
        data_temp_num.fillna(method='pad')
        data_tr_temp = data_tr[data_tr['product_id'] == product_id]
        # print data_tr_temp
        data_temp_num = pd.merge(data_temp_num, data_tr_temp, how='left', on='user_id')
        data_temp_num = data_temp_num.fillna({'predict': 0})
        # print len(data_temp_num), data_temp_num.head()
        del data_temp_num['product_id']
        data_temp_num = data_temp_num.dropna()

        # x = data_temp_num[['count', 'max_order_number', 'mean_order_dow', 'min_order_dow', 'max_order_dow',
        #                    'mean_order_hour_of_day', 'min_order_hour_of_day', 'max_order_hour_of_day',
        #                    'mean_days_since_prior_order', 'min_days_since_prior_order', 'max_days_since_prior_order']]
        x = data_temp_num[['count', 'max_order_number', 'mean_order_dow',
                           'mean_order_hour_of_day',
                           'mean_days_since_prior_order']]
        # x = data_temp_num[['count', 'max_order_number', 'mean_order_dow',
        #                    'mean_order_hour_of_day',
        #                    'mean_days_since_prior_order',
        #                    'count_ord_user', 'count_pro_user', 'mean_order_dow_user',
        #                    'mean_order_hour_of_day_user', 'mean_days_since_prior_order_user']]
        y = data_temp_num['predict']
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
        # print type(y_train)
        i = i + 1
        print i
        print len(data_temp_num), len(x_train), len(x_test)
        # y_train_df = pd.DataFrame(data=y_train)
        predict_ini = 0
        # print data_temp_num[data_temp_num.isnull().values==True]
        if i > 100:
            break
        elif len(data_temp_num) == 1:
            predict_ini = data_temp_num['predict'][0]
            print 'predict: ', predict_ini
        elif (y_train.values.sum() == 0) | (y_train.values.sum() == len(y_train)):
            predict_ini = y_train.values[0]
            print 'predict: ', predict_ini
        else:
            model()
    print u'训练平均准确度：%.2f%%' % (np.mean(train_acc_rate_list))
    print u'测试平均准确度：%.2f%%' % (np.mean(test_acc_rate_list))






