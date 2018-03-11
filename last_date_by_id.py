import pandas as pd
from datetime import datetime


#proba для месяца
def get_test_proba(df, churn_month, path):
    last_month = lambda x: pd.datetime(x.year, x.month, 1)

    df_proba = df.groupby('client_id')[['date','first_prch']].agg(['max','min']).copy()

    # колонка с максимальным месяцем
    df_proba['date']['max'] = df_proba['date']['max'].apply(last_month)
    df_proba['date']['min'] = df_proba['date']['min'].apply(last_month)


    # откидываем людей, которые откинулись до искомого месяца
    df_proba = df_proba[df_proba['date']['max'] >= pd.datetime(2017, churn_month-1, 1)]
    df_proba = df_proba[df_proba['date']['min'] <= pd.datetime(2017, churn_month-1, 1)]

    df_proba['proba'] = df_proba['date']['max'].apply(lambda x: 1 if x == pd.datetime(2017, churn_month-1, 1) else 0)
    df_proba['age'] = df_proba['date']['max'] - df_proba['first_prch']['max']
    df_proba['age'] = df_proba['age'].apply(lambda x: int(x.days/30))

    #df_proba = df_proba.drop(['date']['max','min'],axis = 1)
    print(df_proba.proba.value_counts())
    #df_proba = df_proba.drop([['date']['max']])
    df_proba.to_csv(path)
    return df_proba[['proba','age']]




#############################################
# last_date_by_id - последний месяц для юзера
#############################################
def calc_last_date_by_id():
    df = pd.read_csv('train_data.csv')
    df = df.groupby('id').date.agg(['max'])
    df.to_csv('train_last_date_by_id.csv')

    df = pd.read_csv('test_data.csv')
    df = df.groupby('id').date.agg(['max'])
    df.to_csv('test_last_date_by_id.csv')

def get_last_date_by_id(test=False):
    if test:
        return pd.read_csv('train_last_date_by_id.csv')
    else:
        return pd.read_csv('test_last_date_by_id.csv')

# df = pd.read_csv('train_data.csv',index_col = 'Unnamed: 0')
# df.head(100000).to_csv('train_data_100000.csv')

#############################################
# cat и prod - отбрасываем хвост категориальных прихнаов
#############################################
def add_cat_short_coll(df,col_name='code'):
    code_value_counts_short = long_cols_itms[col_name]
    df[col_name+'_short'] = df[col_name].apply(lambda val: val if val in code_value_counts_short else 'other')
    return df



def parse_date(date_to_parse):
    date = date_to_parse.split('-')
    return datetime(int(date[0]),int(date[1]),int(date[2]))

def discount_bin(x):
    if x > 0:
        return 1
    return 0

def parse_datetime(x):
    return pd.datetime.strptime(x, '%d.%m.%y %H:%M:%S')

def load_data(path):
    df = pd.read_csv(path, index_col = 'Unnamed: 0',
                    dtype = {'time':str,
                             'date':str,
                             'v_l':float,
                             'q':int,
                             'n_tr':int,
                             'sum_b':float,
                             'code_azs':str,
                             'first_prch':str,
                             'location':str,
                             'region':str,
                             'code':str,
                             'code1':str,
                             'percent':float,
                             'type':int},
                    parse_dates=['first_prch'], date_parser = parse_datetime)
    df.columns = ['time','date','petrol_volume','snack_quant','tran_num','sum','azs_id','client_id',
           'first_prch','location','region','item_id','type_id','discount','payment']

    df.date = df.date.apply(parse_date)
    df['used_bonuses'] = df['discount'].apply(discount_bin)

    return df


def one_hot_encode(df,col):
    one_hot = pd.get_dummies(df[col])
    one_hot.columns = [(col+'_'+str(s)) for s in one_hot.columns]
    df = df.drop(col,axis = 1)
    df = df.join(one_hot)
    return df

def add_year_month(df,col_name='date'):
    df['year_month'] = df[col_name].apply(lambda date: date.year*100+date.month)
    return df

def prepare_mega_data_set(sorce='train_data',month=10,suffix='base'):
    #грузим датасет
    df = load_data(sorce+'.csv')

    df_proba = get_test_proba(df,month,sorce+'_proba_'+str(month)+suffix+'.csv')
    df = df[df['date'] < datetime(2017,month,1)]

    # усрать срань
    #drop_cols = ['time','tran_num']
    # числовые колонки
    num_cols = ['petrol_volume','snack_quant','sum','discount','used_bonuses']
    # длинные разряженные классы (чистим от хвостов)
    #long_cols = ['type_id','item_id']
    # классовые
    #cat_cols = ['location', 'region','payment']
    # 'date'
    # 'azs_id'
    # 'first_prch'
    # 'client_id'

    # усрать срань
    for col in drop_cols:
        df = df.drop(col,axis = 1)

    # длинные разряженные классы (чистим от хвостов)
    for col in long_cols:
        df = add_cat_short_coll(df,col)
        df = df.drop(col,axis = 1)
        df = one_hot_encode(df,col+'_short')

    # классовые
    for col in cat_cols:
        df = one_hot_encode(df, col)

    #id транзакции
    df['year_month'] = df['date'].apply(lambda date: date.year*100+date.month)
    df = df.drop('date',axis=1)

    #@TODO!!! - словарик для заправок
    df = df.drop('azs_id',axis=1)


    # группируем по
    gdf = df.groupby(['client_id', 'year_month'])[num_cols].sum()
    print(gdf.head())

    cat_cols = list(set(df.columns.tolist()).difference(set(['client_id', 'year_month']+num_cols)))
    cgdf = df.groupby(['client_id'])[cat_cols].sum()

    #кол-во колонок
    #кол-во новых колонок
    #

    unstacked_df = gdf.unstack()
    unstacked_df.head()

    mega_data_set = pd.merge(unstacked_df, df_proba, left_index=True, right_index=True)
    mega_data_set.head()


    mega_data_set = mega_data_set.fillna(0)
    mega_data_set.to_csv(sorce+'_mega_set_'+str(month)+suffix+'.csv')
    print(mega_data_set.head())


    # обучить 2 модели - на 2 даты: 2017-10 2017-12
    # крос валидация на старатифицированных выборках


drop_cols = ['time', 'tran_num']
# числовые колонки
num_cols = ['petrol_volume', 'snack_quant', 'sum', 'discount', 'used_bonuses']

# классовые
cat_cols = ['location', 'region', 'payment']

# длинные разряженные классы (чистим от хвостов)
long_cols = ['type_id','item_id']
df = pd.read_csv('train_data.csv')
long_cols_itms = {}
code_cv = df['code'].value_counts()
long_cols_itms['item_id'] = code_cv[code_cv > code_cv.mean() / 2].index.tolist()
code_cv = df['code1'].value_counts()
long_cols_itms['type_id'] = code_cv[code_cv > code_cv.mean() / 2].index.tolist()
print(long_cols_itms)

prepare_mega_data_set(sorce='train_data_100000', month=10, suffix='_base_cat')
# prepare_mega_data_set(sorce='train_data', month=12, suffix='_base_cat')
#
# prepare_mega_data_set(sorce='test_data', month=10, suffix='_base_cat')
# prepare_mega_data_set(sorce='test_data', month=12, suffix='_base_cat')