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

    test_y_hat = lr.predict(x_test)
    test_acc_rate = 100*np.mean(test_y_hat == y_test.ravel())
    test_acc_rate_list.append(test_acc_rate)

    y_pred = lr.predict(x_pre)
    return y_pred

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

  #   order_all = pd.read_csv('orders.csv')
  #   order_tr = order_all[order_all['eval_set'] == 'train']
  #   print len(order_tr)
  #   order_tr_pri = order_all[order_all['user_id'].isin(order_tr['user_id'])]
  #   print len(order_tr_pri)
  #
  # # get all prior data for train(test) order
  #   data_tr_pri_list = []
  #   data_all = pd.read_csv('order_products__prior.csv', chunksize=1000000)
  #   for chunk in data_all:
  #       data_tr_pri_list.append(pd.merge(chunk, order_tr_pri, how='inner', on='order_id'))
  #       data_tr_pri = pd.concat(data_tr_pri_list, ignore_index=True)
  #   print len(data_tr_pri)
  #   del data_all
  #
  #   # get product in train order to set predict=1(target)
  #   order_tr2 = order_all[order_all['eval_set'] == 'train']
  #   data_tr = pd.read_csv('order_products__train.csv')
  #   data_tr = pd.merge(data_tr, order_tr2, how='left', on='order_id')
  #   data_tr['predict'] = 1
  #   data_tr = data_tr.loc[:, ['user_id', 'product_id', 'predict']]
  #
  #   order_te = order_all[order_all['eval_set'] == 'test']
  #   order_te_pri = order_all[order_all['user_id'].isin(order_te['user_id'])]
  #   print len(order_te_pri)
  #   # get all prior data for train(test) order
  #   data_te_pri_list = []
  #   data_all = pd.read_csv('order_products__prior.csv', chunksize=1000000)
  #   for chunk in data_all:
  #       data_te_pri_list.append(pd.merge(chunk, order_te_pri, how='inner', on='order_id'))
  #       data_te_pri = pd.concat(data_te_pri_list, ignore_index=True)
  #   print len(data_te_pri)
  #   del data_all
  #
  #   data_tr_pri.to_csv('data_tr_pri.csv', index=False, sep=',')
  #   data_te_pri.to_csv('data_te_pri.csv', index=False, sep=',')
  #   data_tr.to_csv('data_tr.csv', index=False, sep=',')
  #   order_te.to_csv('order_te.csv', index=False, sep=',')

    # # read temp data
    # data_tr_pri = pd.read_csv('data_tr_pri.csv')
    # print len(data_tr_pri)
    # data_te_pri = pd.read_csv('data_te_pri.csv')
    # print len(data_te_pri)
    # data_tr = pd.read_csv('data_tr.csv')
    # print len(data_tr)
    order_te = pd.read_csv('order_te.csv')
    print len(order_te)
    #
    # # get the former product for both train and test order
    # data_tr_pro_list = data_tr_pri.product_id.unique()
    # print len(data_tr_pro_list)
    # data_te_pro_list = data_te_pri.product_id.unique()
    # print len(data_te_pro_list)
    # data_te_pro_list = np.intersect1d(data_tr_pro_list, data_te_pro_list)
    # print len(data_te_pro_list)
    #
    # train_acc_rate_list = []
    # test_acc_rate_list = []
    # y_pre_list = pd.DataFrame()
    # i = 0
    # list_n = len(data_te_pro_list) - 42000 - 1
    # start_n = 42001
    # data_te_pro_list_2 = np.array([0]*list_n)
    # for j in range(list_n):
    #     print (j + start_n)
    #     data_te_pro_list_2[j] = data_te_pro_list[j + start_n]
    #     # print data_te_pro_list_2[j]
    # data_te_pro_list = data_te_pro_list_2
    # print len(data_te_pro_list)
    # for product_id in data_te_pro_list:
    #     data_temp = data_tr_pri[data_tr_pri['product_id'] == product_id]
    #     data_temp['count'] = data_temp['order_id']
    #     data_temp['max_order_number'] = data_temp['order_number']
    #     data_temp['mean_order_dow'] = data_temp['order_dow']
    #     data_temp['mean_order_hour_of_day'] = data_temp['order_hour_of_day']
    #     data_temp['mean_days_since_prior_order'] = data_temp['days_since_prior_order']
    #     data_temp_num = data_temp.groupby('user_id').agg({'count': pd.Series.nunique,
    #                                                       'max_order_number': np.max,
    #                                                       'mean_order_dow': np.mean,
    #                                                       'mean_order_hour_of_day': np.mean,
    #                                                       'mean_days_since_prior_order': np.mean})
    #     data_temp_num['user_id'] = data_temp_num.index
    #     # print data_temp_num.head()
    #     # break
    #     data_temp_num.fillna(method='pad')
    #     data_tr_temp = data_tr[data_tr['product_id'] == product_id]
    #     # print data_tr_temp
    #     data_temp_num = pd.merge(data_temp_num, data_tr_temp, how='left', on='user_id')
    #     data_temp_num = data_temp_num.fillna({'predict': 0})
    #     # print len(data_temp_num), data_temp_num.head()
    #     del data_temp_num['product_id']
    #     data_temp_num = data_temp_num.dropna()
    #
    #     data_temp = data_te_pri[data_te_pri['product_id'] == product_id]
    #     data_temp['count'] = data_temp['order_id']
    #     data_temp['max_order_number'] = data_temp['order_number']
    #     data_temp['mean_order_dow'] = data_temp['order_dow']
    #     data_temp['mean_order_hour_of_day'] = data_temp['order_hour_of_day']
    #     data_temp['mean_days_since_prior_order'] = data_temp['days_since_prior_order']
    #     data_temp_num_te = data_temp.groupby('user_id').agg({'count': pd.Series.nunique,
    #                                                       'max_order_number': np.max,
    #                                                       'mean_order_dow': np.mean,
    #                                                       'mean_order_hour_of_day': np.mean,
    #                                                       'mean_days_since_prior_order': np.mean,
    #                                                       'product_id': np.min})
    #     data_temp_num_te['user_id'] = data_temp_num_te.index
    #     data_temp_num_te = data_temp_num_te.fillna(method='pad')
    #     data_temp_num_te = data_temp_num_te.dropna()
    #
    #     x = data_temp_num[['count', 'max_order_number', 'mean_order_dow',
    #                        'mean_order_hour_of_day',
    #                        'mean_days_since_prior_order']]
    #
    #     y = data_temp_num['predict']
    #     del data_temp_num
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    #
    #     x_pre_user = data_temp_num_te[['user_id', 'product_id', 'count', 'max_order_number', 'mean_order_dow',
    #                               'mean_order_hour_of_day',
    #                               'mean_days_since_prior_order']]
    #     x_pre = x_pre_user[['count', 'max_order_number', 'mean_order_dow',
    #                               'mean_order_hour_of_day',
    #                               'mean_days_since_prior_order']]
    #
    #     del data_temp_num_te
    #     i = i + 1
    #     print i
    #     print len(y), len(x_train), len(x_test), len(x_pre)
    #     # y_train_df = pd.DataFrame(data=y_train)
    #     y_pre = []
    #     y_pre2 = []
    #     y_pre3 = []
    #     # print data_temp_num[data_temp_num.isnull().values==True]
    #     if len(x_pre) == 0:
    #         continue
    #     else:
    #          if len(y) == 0:
    #              x_pre_user_temp = x_pre_user[['user_id', 'product_id']]
    #              x_pre_user_temp['predict'] = 0
    #              # print x_pre_user_temp
    #          elif len(y) == 1:
    #              y_pre2 = y.values[0]
    #              # print 'predict: ', y_pre2
    #              x_pre_user_temp = x_pre_user[['user_id', 'product_id']]
    #              x_pre_user_temp['predict'] = y_pre2
    #              # print x_pre_user_temp
    #              # y_pre_list = np.append(y_pre_list, [product_id, y_pre2], axis=0)
    #              # print y_pre_list
    #          elif (y_train.values.sum() == 0) | (y_train.values.sum() == len(y_train)):
    #              y_pre3 = y_train.values[0]
    #              # print 'predict: ', y_pre3
    #              x_pre_user_temp = x_pre_user[['user_id', 'product_id']]
    #              x_pre_user_temp['predict'] = y_pre3
    #              # print x_pre_user_temp
    #              # y_pre_list = np.append(y_pre_list, [product_id, y_pre3], axis=0)
    #              # print y_pre_list
    #          else:
    #              y_pre = model()
    #              x_pre_user_temp = x_pre_user[['user_id', 'product_id']]
    #              x_pre_user_temp['predict'] = y_pre
    #              # print x_pre_user_temp
    #              # for y_pre_e in y_pre:
    #              #     y_pre_list = np.append(y_pre_list, [product_id, y_pre_e], axis=0)
    #              # print y_pre, y_pre_list
    #          y_pre_list = pd.concat([y_pre_list, x_pre_user_temp], ignore_index=True, axis=0)
    # # print y_pre_list
    # y_pre_list = y_pre_list[y_pre_list['predict'] == 1]
    # y_pre_list = pd.merge(y_pre_list, order_te, how='left', on='user_id')
    # y_pre_list = y_pre_list[['order_id', 'product_id']]
    # # y_pre_list.to_csv('y_pre_list.csv', index=False, sep=',')
    # y_pre_list.to_csv('y_pre_list.csv', index=False, sep=',', mode='a', header=False)

    y_pre_list = pd.read_csv('y_pre_list.csv')
    #y_pre_list = y_pre_list.sort(['order_id', 'product_id'])
    y_pre_list = y_pre_list.sort_values(['order_id', 'product_id'])
    # print y_pre_list
    submission = pd.DataFrame()
    for order_id in y_pre_list.order_id.unique():
        products = ''
        for product_id in y_pre_list[y_pre_list['order_id'] == order_id].product_id.unique():
            products = ' '.join([products, '%s'%(product_id)])
        # print order_id, products
        submission_temp = pd.DataFrame(data = np.array([order_id, products]).reshape(-1, 2), columns=['order_id', 'products'])
        submission = pd.concat([submission, submission_temp], ignore_index=True, axis=0)
    # submission.sort(['order_id'])
    print submission
    submission_merge = pd.merge(order_te, submission, how='left', on='order_id')
    print submission_merge
    submission = submission_merge[['order_id', 'products']]
    submission.to_csv('submission.csv', index=False, sep=',')

    # print u'训练平均准确度：%.2f%%' % (np.mean(train_acc_rate_list))
    # print u'测试平均准确度：%.2f%%' % (np.mean(test_acc_rate_list))
