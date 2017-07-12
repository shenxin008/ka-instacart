#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import svm
import math


def split_train_test(text_df, size=0.5):
    """
        分割训练集和测试集
    """
    # 为保证每个类中的数据能在训练集中和测试集中的比例相同，所以需要依次对每个类进行处理
    train_text_df = pd.DataFrame()
    test_text_df = pd.DataFrame()

    labels = [0, 1]
    for label in labels:
        # 找出label的记录
        text_df_w_label = text_df[text_df['predict'] == label]
        # 重新设置索引，保证每个类的记录是从0开始索引，方便之后的拆分
        text_df_w_label = text_df_w_label.reset_index()

        # 默认按80%训练集，20%测试集分割
        # 这里为了简化操作，取前80%放到训练集中，后20%放到测试集中
        # 当然也可以随机拆分80%，20%（尝试实现下DataFrame中的随机拆分）

        # 该类数据的行数
        n_lines = text_df_w_label.shape[0]
        split_line_no = int(math.floor(n_lines * size))
        text_df_w_label_train = text_df_w_label.iloc[:split_line_no, :]
        text_df_w_label_test = text_df_w_label.iloc[split_line_no:, :]

        # 放入整体训练集，测试集中
        train_text_df = train_text_df.append(text_df_w_label_train)
        test_text_df = test_text_df.append(text_df_w_label_test)

    train_text_df = train_text_df.reset_index()
    test_text_df = test_text_df.reset_index()
    return train_text_df, test_text_df


def model():
    # lr = Pipeline([('sc', StandardScaler()),
    #                ('poly', PolynomialFeatures(degree=1)),
    #                ('clf', LogisticRegression())])

    # lr = DecisionTreeClassifier(criterion='entropy')
    lr = RandomForestClassifier(n_estimators=200, criterion='entropy', class_weight={0: 0.1, 1: 1})
    # lr = svm.SVC(C=1, kernel='linear')
    # lr = svm.SVC(C=0.8, kernel='rbf', gamma=10, class_weight={0: 1, 1: 10})
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
    x_all_pred = lr.predict(x)
    all_acc_rate = 100*np.mean(x_all_pred == y.ravel())
    all_acc_rate_list.append(all_acc_rate)
    x_all_len.append(len(x))

    y_pred = lr.predict(x_pre)
    # print y.values.mean(), np.mean(x_all_pred)
    print y_test.values.mean(), np.mean(test_y_hat)
    # print 'test_y_hat', test_y_hat
    print 'y_pred', y_pred

    return y_pred, x_all_pred

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
    # order_pri_tr.to_csv('order_pri_tr.csv', index=False, sep=',')
    # order_pri_te.to_csv('order_pri_te.csv', index=False, sep=',')

    # read temp data
    data_pri_tr = pd.read_csv('data_pri_tr.csv')
    print len(data_pri_tr)
    data_pri_te = pd.read_csv('data_pri_te.csv')
    print len(data_pri_te)
    order_tr = pd.read_csv('order_tr.csv')
    print len(order_tr)
    order_te = pd.read_csv('order_te.csv')
    print len(order_te)
    # order_pri_tr = pd.read_csv('order_pri_tr.csv')
    # print len(order_pri_tr)
    # order_pri_te = pd.read_csv('order_pri_te.csv')
    # print len(order_pri_te)

    data_tr = pd.read_csv('order_products__train.csv')
    # data_tr_re = data_tr[data_tr['reordered'] == 1]

    pri_tr_pro_list = data_pri_tr.product_id.unique()
    pri_te_pro_list = data_pri_te.product_id.unique()
    pro_list = np.intersect1d(pri_tr_pro_list, pri_te_pro_list)
    print len(pri_tr_pro_list), len(pri_te_pro_list), len(pro_list)

    # data_pri_tr_last_temp = order_pri_tr.groupby('user_id')[['order_number']].max()
    # data_pri_tr_last_temp['user_id'] = data_pri_tr_last_temp.index
    # order_pri_tr_last = pd.merge(data_pri_tr_last_temp, order_pri_tr, how='left', on=['user_id', 'order_number'])
    # order_pri_tr_last = order_pri_tr_last[['order_number', 'order_id']]
    # data_pri_tr_last = pd.merge(order_pri_tr_last, data_pri_tr, how='left', on='order_id')
    # data_pri_tr_last = data_pri_tr_last[['user_id', 'product_id']]
    # data_pri_tr_last['last_buy'] = 1
    # print data_pri_tr_last
    #
    # data_pri_te_last_temp = order_pri_te.groupby('user_id')[['order_number']].max()
    # data_pri_te_last_temp['user_id'] = data_pri_te_last_temp.index
    # order_pri_te_last = pd.merge(data_pri_te_last_temp, order_pri_te, how='left', on=['user_id', 'order_number'])
    # order_pri_te_last = order_pri_te_last[['order_number', 'order_id']]
    # data_pri_te_last = pd.merge(order_pri_te_last, data_pri_te, how='left', on='order_id')
    # data_pri_te_last = data_pri_te_last[['user_id', 'product_id']]
    # data_pri_te_last['last_buy'] = 1





    train_acc_rate_list = []
    test_acc_rate_list = []
    all_acc_rate_list = []
    x_train_len = []
    x_test_len = []
    x_all_len = []
    y_pre_list = pd.DataFrame()
    x_pre_list = pd.DataFrame()
    x_re_list = pd.DataFrame()
    i = 0
    pro_list = pro_list[0: 50]
    for product_id in pro_list:
        i += 1
        print i, product_id
        data_before_tr_temp = data_pri_tr[data_pri_tr['product_id'] == product_id]
        data_before_tr_temp['count'] = data_before_tr_temp['order_id']
        data_before_tr_temp['max_order_number'] = data_before_tr_temp['order_number']
        data_before_tr_temp['mean_order_dow'] = data_before_tr_temp['order_dow']
        data_before_tr_temp['mean_order_hour_of_day'] = data_before_tr_temp['order_hour_of_day']
        data_before_tr_temp['mean_days_since_prior_order'] = data_before_tr_temp['days_since_prior_order']
        data_before_tr_temp['count_reordered'] = data_before_tr_temp['reordered']
        data_before_tr_temp = data_before_tr_temp.fillna({'mean_days_since_prior_order': 0})
        data_before_tr_temp_2 = data_before_tr_temp.groupby('user_id').agg({'count': pd.Series.nunique,
                                                          'max_order_number': np.max,
                                                          'mean_order_dow': np.mean,
                                                          'mean_order_hour_of_day': np.mean,
                                                          'mean_days_since_prior_order': np.mean,
                                                          'product_id': np.min,
                                                          'count_reordered': np.sum})
        data_before_tr_temp_2['user_id'] = data_before_tr_temp_2.index
        data_before_tr_temp_2 = pd.merge(data_before_tr_temp_2, order_tr, how='left', on='user_id')
        # data_pri_tr_last = data_pri_tr_last[data_pri_tr_last['product_id'] == product_id]
        # data_pri_tr_last = data_pri_tr_last[['user_id', 'last_buy']]
        # data_before_tr_temp_2 = pd.merge(data_before_tr_temp_2, data_pri_tr_last, how='left', on='user_id')
        # data_before_tr_temp_2 = data_before_tr_temp_2.fillna({'last_buy': 0})
        data_today_tr_temp = data_tr[data_tr['product_id'] == product_id]
        # data_today_tr_temp['predict'] = 1
        data_today_tr_temp['predict'] = data_today_tr_temp['reordered']
        data_today_tr_temp.drop(['product_id', 'add_to_cart_order', 'reordered'], axis=1, inplace=True)
        data_before_tr_temp_2 = pd.merge(data_before_tr_temp_2, data_today_tr_temp, how='left', on='order_id')
        data_before_tr_temp_2 = data_before_tr_temp_2.fillna({'predict': 0})
        # print data_before_tr_temp_2.predict.sum(), data_before_tr_temp_2.reordered.sum()
        data_before_tr_temp_2 = data_before_tr_temp_2.dropna()
        # x = data_before_tr_temp_2[['count', 'max_order_number', 'mean_order_dow', 'mean_order_hour_of_day', 'mean_days_since_prior_order'
        #                            , 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']]
        x = data_before_tr_temp_2[
            ['count', 'max_order_number', 'mean_order_dow', 'mean_order_hour_of_day', 'mean_days_since_prior_order']]
        y = data_before_tr_temp_2['predict']
        # xy = data_before_tr_temp_2[['predict', 'count', 'max_order_number', 'mean_order_dow', 'mean_order_hour_of_day', 'mean_days_since_prior_order'
        #                                 , 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']]
        xy = data_before_tr_temp_2[['predict', 'count', 'max_order_number', 'mean_order_dow', 'mean_order_hour_of_day',
                                    'mean_days_since_prior_order']]
        x_user = data_before_tr_temp_2[['user_id', 'product_id', 'order_id']]

        data_before_tr_temp = data_pri_te[data_pri_te['product_id'] == product_id]
        data_before_tr_temp['count'] = data_before_tr_temp['order_id']
        data_before_tr_temp['max_order_number'] = data_before_tr_temp['order_number']
        data_before_tr_temp['mean_order_dow'] = data_before_tr_temp['order_dow']
        data_before_tr_temp['mean_order_hour_of_day'] = data_before_tr_temp['order_hour_of_day']
        data_before_tr_temp['mean_days_since_prior_order'] = data_before_tr_temp['days_since_prior_order']
        data_before_tr_temp['count_reordered'] = data_before_tr_temp['reordered']
        data_before_tr_temp = data_before_tr_temp.fillna({'mean_days_since_prior_order': 0})
        data_before_tr_temp_2 = data_before_tr_temp.groupby('user_id').agg({'count': pd.Series.nunique,
                                                                            'max_order_number': np.max,
                                                                            'mean_order_dow': np.mean,
                                                                            'mean_order_hour_of_day': np.mean,
                                                                            'mean_days_since_prior_order': np.mean,
                                                                            'product_id': np.min,
                                                                            'count_reordered': np.sum})
        data_before_tr_temp_2['user_id'] = data_before_tr_temp_2.index
        data_before_tr_temp_2 = pd.merge(data_before_tr_temp_2, order_te, how='left', on='user_id')
        # data_pri_te_last = data_pri_te_last[data_pri_te_last['product_id'] == product_id]
        # data_pri_te_last = data_pri_te_last[['user_id', 'last_buy']]
        # data_before_tr_temp_2 = pd.merge(data_before_tr_temp_2, data_pri_te_last, how='left', on='user_id')
        # data_before_tr_temp_2 = data_before_tr_temp_2.fillna({'last_buy': 0})
        # x_pre = data_before_tr_temp_2[['count', 'max_order_number', 'mean_order_dow', 'mean_order_hour_of_day', 'mean_days_since_prior_order'
        #                            , 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']]
        x_pre = data_before_tr_temp_2[['count', 'max_order_number', 'mean_order_dow', 'mean_order_hour_of_day', 'mean_days_since_prior_order']]
        # print 'x_pre', x_pre
        x_pre_user = data_before_tr_temp_2[['user_id', 'product_id', 'order_id']]

        if len(xy) > 3:
            # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
            x_train, x_test = split_train_test(xy)
            # print xy.head(),  x_train.head()
            y_train = x_train['predict']
            x_train.drop(['predict', 'level_0', 'index'], axis=1, inplace=True)
            # print x_train.head()
            # break
            y_test = x_test['predict']
            x_test.drop(['predict', 'level_0', 'index'], axis=1, inplace=True)
            print len(x_train), len(x_test), len(x_pre_user)
            print np.sum(y_train), np.sum(y_test)
            if (y_train.values.sum() != 0) & (y_train.values.sum() != len(y_train)):
                y_pre, x_all_pre = model()
            else:
                y_pre = np.array([y_train.values[0]] * len(x_pre_user))
                x_all_pre = np.array([y_train.values[0]] * len(xy))
                print 'all same', product_id, y_train.values, y_pre
            x_pre_user['predict'] = y_pre
            x_user['predict'] = x_all_pre
        elif len(xy) == 3:
            y_pre = np.array(np.floor(y_train.values.sum() / 2) * len(x_pre_user))
            x_all_pre = np.array(np.floor(y_train.values.sum() / 2) * len(xy))
            print '=3', y_pre
            x_pre_user['predict'] = y_pre
            x_user['predict'] = x_all_pre
        elif len(xy) == 2:
            if y_train.values.sum() == 0:
                y_pre = np.array([0] * len(x_pre_user))
                x_all_pre = np.array([0] * len(xy))
            else:
                y_pre = np.array([1] * len(x_pre_user))
                x_all_pre = np.array([1] * len(xy))
            print '=2', y_pre
            x_pre_user['predict'] = y_pre
            x_user['predict'] = x_all_pre
        else:
            y_pre = np.array(y_train.values[0])
            x_all_pre = np.array(y_train.values[0])
            print '=1', y_pre
            x_pre_user['predict'] = y_pre
            x_user['predict'] = x_all_pre
        print len(x_pre_user), x_pre_user['predict'].sum()
        y_pre_list = pd.concat([y_pre_list, x_pre_user], ignore_index=True, axis=0)
        x_pre_list = pd.concat([x_pre_list, x_user], ignore_index=True, axis=0)
        x_re = data_tr[data_tr['product_id'] == product_id]
        x_re_list = pd.concat([x_re_list, x_re], ignore_index=True, axis=0)

    print len(y_pre_list), y_pre_list['predict'].sum()
    y_pre_list = y_pre_list[y_pre_list['predict'] == 1]
    y_pre_list = y_pre_list[['order_id', 'product_id']]
    y_pre_list = y_pre_list.sort_values(['order_id', 'product_id'])

    print len(x_pre_list), x_pre_list['predict'].sum()
    x_pre_list = x_pre_list[x_pre_list['predict'] == 1]
    x_pre_list = x_pre_list[['order_id', 'product_id']]
    x_pre_list = x_pre_list.sort_values(['order_id', 'product_id'])

    x_all_list = x_re_list.order_id.unique()
    submission_all = pd.DataFrame({'order_id': x_all_list, 'nomeaning': np.arange(len(x_all_list))})

    x_re_list = x_re_list[x_re_list['reordered'] == 1]
    x_re_list = x_re_list[['order_id', 'product_id']]
    x_re_list = x_re_list.sort_values(['order_id', 'product_id'])

    # submission = pd.DataFrame()
    # for order_id in y_pre_list.order_id.unique():
    #     products = ''
    #     for product_id in y_pre_list[y_pre_list['order_id'] == order_id].product_id.unique():
    #         products = str.strip(' '.join([products, '%s' % (product_id)]))
    #     submission_temp = pd.DataFrame({'order_id': pd.Series(order_id), 'products': pd.Series(products)})
    #     submission = pd.concat([submission, submission_temp], ignore_index=True, axis=0)
    # print len(submission), submission.head()
    # submission_merge = pd.merge(order_te, submission, how='left', on='order_id')
    # submission = submission_merge[['order_id', 'products']]
    # submission.to_csv('submission.csv', index=False, sep=',')

    submission_x = pd.DataFrame()
    for order_id in x_pre_list.order_id.unique():
        products = ''
        for product_id in x_pre_list[x_pre_list['order_id'] == order_id].product_id.unique():
            products = str.strip(' '.join([products, '%s' % (product_id)]))
        submission_temp = pd.DataFrame({'order_id': pd.Series(order_id), 'products': pd.Series(products)})
        submission_x = pd.concat([submission_x, submission_temp], ignore_index=True, axis=0)
    print len(submission_x), submission_x.head()
    # submission_merge = pd.merge(order_tr, submission, how='left', on='order_id')
    # submission_x = submission_merge[['order_id', 'products']]
    # submission_x.to_csv('submission_x.csv', index=False, sep=',')

    submission_x_re = pd.DataFrame()
    for order_id in x_re_list.order_id.unique():
        products = ''
        for product_id in x_re_list[x_re_list['order_id'] == order_id].product_id.unique():
            products = str.strip(' '.join([products, '%s' % (product_id)]))
        submission_temp = pd.DataFrame({'order_id': pd.Series(order_id), 'products': pd.Series(products)})
        submission_x_re = pd.concat([submission_x_re, submission_temp], ignore_index=True, axis=0)
    print len(submission_x_re), submission_x_re.head()

    submission_x.rename(columns={'products': 'products_pre'}, inplace=True)
    submission_all = pd.merge(submission_all, submission_x_re, how='left', on='order_id')
    submission_all = pd.merge(submission_all, submission_x, how='left', on='order_id')
    print len(submission_all), submission_all
    submission_all.to_csv('submission_all.csv', index=False, sep=',')

    print u'训练平均准确度：%.2f%%' % (np.average(train_acc_rate_list, weights=x_train_len))
    print u'测试平均准确度：%.2f%%' % (np.average(test_acc_rate_list, weights=x_test_len))
    print u'测试平均准确度：%.2f%%' % (np.average(all_acc_rate_list, weights=x_all_len))

    f_list = []
    for order_id in x_all_list:
        x_re_temp = x_re_list[x_re_list['order_id'] == order_id]
        x_pre_temp = x_pre_list[x_pre_list['order_id'] == order_id]
        if (len(x_re_temp) + len(x_pre_temp)) == 0:
            f = 1
        elif len(x_re_temp) == 0:
            f = 0
        elif len(x_pre_temp) == 0:
            f = 0
        else:
            same = pd.merge(x_re_temp, x_pre_temp, how='inner', on='product_id')
            p = len(same) / len(x_pre_temp)
            r = len(same) / len(x_re_temp)
            f = (2 * p * r) / (p + r)
        f_list.append(f)
    mfs = np.mean(f_list)
    print mfs
