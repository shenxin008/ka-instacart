#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

def model():
    lr = Pipeline([('sc', StandardScaler()),
                   ('poly', PolynomialFeatures(degree=1)),
                   ('clf', LogisticRegression())])
    lr.fit(x_train, y_train.ravel())

    y_hat = lr.predict(x_train)
    train_acc_rate = 100*np.mean(y_hat == y_train.ravel())
    train_acc_rate_list.append(train_acc_rate)
    x_train_len.append(len(x_train))
    # print train_acc_rate

    test_y_hat = lr.predict(x_test)
    test_acc_rate = 100*np.mean(test_y_hat == y_test.ravel())
    test_acc_rate_list.append(test_acc_rate)
    x_test_len.append(len(x_test))
    # print test_acc_rate

    y_pred = lr.predict(x_pre)
    return y_pred

if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    # order_all = pd.read_csv('orders.csv')
    # order_tr = order_all[order_all['eval_set'] == 'train']
    # order_te = order_all[order_all['eval_set'] == 'test']
    # order_pri = order_all[order_all['eval_set'] == 'prior']
    # order_pri_tr = order_pri[order_pri['user_id'].isin(order_tr['user_id'])]
    # order_pri_te = order_pri[order_pri['user_id'].isin(order_te['user_id'])]
    # print len(order_pri), len(order_pri_tr), len(order_pri_te)
    #
    # data_pri = pd.read_csv('order_products__prior.csv')
    # data_pri_tr = pd.merge(data_pri, order_pri_tr, how='inner', on='order_id')
    # data_pri_te = pd.merge(data_pri, order_pri_te, how='inner', on='order_id')
    # del data_pri
    #
    # data_pri_tr.to_csv('data_pri_tr.csv', index=False, sep=',')
    # data_pri_te.to_csv('data_pri_te.csv', index=False, sep=',')
    # order_tr.to_csv('order_tr.csv', index=False, sep=',')
    # order_te.to_csv('order_te.csv', index=False, sep=',')

    # read temp data
    data_pri_tr = pd.read_csv('data_pri_tr.csv')
    print len(data_pri_tr)
    data_pri_te = pd.read_csv('data_pri_te.csv')
    print len(data_pri_te)
    order_tr = pd.read_csv('order_tr.csv')
    print len(order_tr)
    order_te = pd.read_csv('order_te.csv')
    print len(order_te)

    pri_tr_pro_list = data_pri_tr.product_id.unique()
    pri_te_pro_list = data_pri_te.product_id.unique()
    pro_list = np.intersect1d(pri_tr_pro_list, pri_te_pro_list)
    print len(pri_tr_pro_list), len(pri_te_pro_list), len(pro_list)

    data_tr = pd.read_csv('order_products__train.csv')

    train_acc_rate_list = []
    test_acc_rate_list = []
    x_train_len = []
    x_test_len = []
    y_pre_list = pd.DataFrame()
    i = 0
    pro_list = pro_list[0: 50]
    for product_id in pro_list:
        i += 1
        print i
        data_before_tr_temp = data_pri_tr[data_pri_tr['product_id'] == product_id]
        data_before_tr_temp['count'] = data_before_tr_temp['order_id']
        data_before_tr_temp['max_order_number'] = data_before_tr_temp['order_number']
        data_before_tr_temp['mean_order_dow'] = data_before_tr_temp['order_dow']
        data_before_tr_temp['mean_order_hour_of_day'] = data_before_tr_temp['order_hour_of_day']
        data_before_tr_temp['mean_days_since_prior_order'] = data_before_tr_temp['days_since_prior_order']
        data_before_tr_temp = data_before_tr_temp.fillna({'mean_days_since_prior_order': 0})
        data_before_tr_temp_2 = data_before_tr_temp.groupby('user_id').agg({'count': pd.Series.nunique,
                                                          'max_order_number': np.max,
                                                          'mean_order_dow': np.mean,
                                                          'mean_order_hour_of_day': np.mean,
                                                          'mean_days_since_prior_order': np.mean})
        data_before_tr_temp_2['user_id'] = data_before_tr_temp_2.index
        data_before_tr_temp_2 = pd.merge(data_before_tr_temp_2, order_tr, how='left', on='user_id')
        data_today_tr_temp = data_tr[data_tr['product_id'] == product_id]
        data_today_tr_temp['predict'] = 1
        data_today_tr_temp.drop(['product_id', 'add_to_cart_order', 'reordered'], axis=1, inplace=True)
        data_before_tr_temp_2 = pd.merge(data_before_tr_temp_2, data_today_tr_temp, how='left', on='order_id')
        data_before_tr_temp_2 = data_before_tr_temp_2.fillna({'predict': 0})
        data_before_tr_temp_2 = data_before_tr_temp_2.dropna()
        x = data_before_tr_temp_2[['count', 'max_order_number', 'mean_order_dow', 'mean_order_hour_of_day', 'mean_days_since_prior_order'
                                   , 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']]
        y = data_before_tr_temp_2['predict']

        data_before_tr_temp = data_pri_te[data_pri_te['product_id'] == product_id]
        data_before_tr_temp['count'] = data_before_tr_temp['order_id']
        data_before_tr_temp['max_order_number'] = data_before_tr_temp['order_number']
        data_before_tr_temp['mean_order_dow'] = data_before_tr_temp['order_dow']
        data_before_tr_temp['mean_order_hour_of_day'] = data_before_tr_temp['order_hour_of_day']
        data_before_tr_temp['mean_days_since_prior_order'] = data_before_tr_temp['days_since_prior_order']
        data_before_tr_temp = data_before_tr_temp.fillna({'mean_days_since_prior_order': 0})
        data_before_tr_temp_2 = data_before_tr_temp.groupby('user_id').agg({'count': pd.Series.nunique,
                                                                            'max_order_number': np.max,
                                                                            'mean_order_dow': np.mean,
                                                                            'mean_order_hour_of_day': np.mean,
                                                                            'mean_days_since_prior_order': np.mean,
                                                                            'product_id': np.min})
        data_before_tr_temp_2['user_id'] = data_before_tr_temp_2.index
        data_before_tr_temp_2 = pd.merge(data_before_tr_temp_2, order_te, how='left', on='user_id')
        x_pre = data_before_tr_temp_2[['count', 'max_order_number', 'mean_order_dow', 'mean_order_hour_of_day', 'mean_days_since_prior_order'
                                   , 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']]
        x_pre_user = data_before_tr_temp_2[['user_id', 'product_id', 'order_id']]

        if len(x) > 3:
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
            print len(x_train), len(x_test), len(x_pre_user)
            if (y_train.values.sum() != 0) & (y_train.values.sum() != len(y_train)):
                y_pre = model()
            else:
                y_pre = np.array([y_train.values[0]] * len(x_pre_user))
                print y_pre
            x_pre_user['predict'] = y_pre
        else:
            y_pre = np.array(np.floor(y_train.values.sum() / 2) * len(x_pre_user))
            print len(x_pre_user), y_pre
            x_pre_user['predict'] = y_pre
        y_pre_list = pd.concat([y_pre_list, x_pre_user], ignore_index=True, axis=0)
    print len(y_pre_list), y_pre_list['predict'].sum()
    y_pre_list = y_pre_list[y_pre_list['predict'] == 1]
    y_pre_list = y_pre_list[['order_id', 'product_id']]
    y_pre_list = y_pre_list.sort_values(['order_id', 'product_id'])

    submission = pd.DataFrame()
    for order_id in y_pre_list.order_id.unique():
        products = ''
        for product_id in y_pre_list[y_pre_list['order_id'] == order_id].product_id.unique():
            products = str.strip(' '.join([products, '%s' % (product_id)]))
        submission_temp = pd.DataFrame({'order_id': pd.Series(order_id), 'products': pd.Series(products)})
        submission = pd.concat([submission, submission_temp], ignore_index=True, axis=0)
    print len(submission), submission.head()
    submission_merge = pd.merge(order_te, submission, how='left', on='order_id')
    submission = submission_merge[['order_id', 'products']]
    submission.to_csv('submission.csv', index=False, sep=',')

    print u'训练平均准确度：%.2f%%' % (np.average(train_acc_rate_list, weights=x_train_len))
    print u'测试平均准确度：%.2f%%' % (np.average(test_acc_rate_list, weights=x_test_len))



