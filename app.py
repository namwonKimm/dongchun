from flask import Flask, render_template
from sklearn.ensemble import RandomForestClassifier
import pandas as pd #기술통계량
from flask import request
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    global rf

    store_member = pd.read_csv('/home/namwon/datas/store_member_lca_final.csv')
    gradeCount = []

    #################### 랜덤포레스트 모델 수행 ###################

    store_member_clean = store_member.copy()
    store_member_clean.drop(['회원번호','생년','월별 평균 방문횟수_단골','방문당 평균 구매금액_소고','마지막방문날짜','1달 이상 미방문 고객','현재-마지막','주소', '구매금액', '회원등급', '방문횟수', '방문당 평균 구매금액', '월별 평균 방문횟수', 'k-means군집','Unnamed: 0'], axis=1, inplace=True)

    store_member_dummy = pd.get_dummies(store_member_clean)
    np.random.seed(seed=1234)

    store_member_dummy_y = store_member_dummy["lca"]
    store_member_dummy_x = store_member_dummy.drop("lca", axis=1, inplace=False)

    rf = RandomForestClassifier(criterion='gini', max_depth=6, max_features=None, min_samples_leaf=7, n_estimators=8, random_state=1234)
    rf.fit(store_member_dummy_x, store_member_dummy_y)

    ############################################################

    vip = store_member[store_member['회원등급'] == 'VIP']['회원등급'].count()
    gold = store_member[store_member['회원등급'] == 'GOLD']['회원등급'].count()
    silver = store_member[store_member['회원등급'] == 'SILVER']['회원등급'].count()
    bronze = store_member[store_member['회원등급'] == 'BRONZE']['회원등급'].count()
    normal = store_member[store_member['회원등급'] == 'NORMAL']['회원등급'].count()

    gradeCount.append(vip)
    gradeCount.append(gold)
    gradeCount.append(silver)
    gradeCount.append(bronze)
    gradeCount.append(normal)

    store_profit = pd.read_csv('/home/namwon/datas/store_profit.csv')
    profitPerMarket = []
    profitPerMarket.append(int(store_profit[store_profit['매장'] == '매장1']['매장별매출']))
    profitPerMarket.append(int(store_profit[store_profit['매장'] == '매장2']['매장별매출']))
    profitPerMarket.append(int(store_profit[store_profit['매장'] == '매장3']['매장별매출']))
    profitPerMarket.append(int(store_profit[store_profit['매장'] == '매장4']['매장별매출']))

    store_profit_ex = store_profit.drop('매장별매출', axis=1)

    market1ProfitPerMonth = []
    market2ProfitPerMonth = []
    market3ProfitPerMonth = []
    market4ProfitPerMonth = []

    market1ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장1']['1월매출']))
    market1ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장1']['2월매출']))
    market1ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장1']['3월매출']))
    market1ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장1']['4월매출']))
    market1ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장1']['5월매출']))
    market1ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장1']['6월매출']))

    market2ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장2']['1월매출']))
    market2ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장2']['2월매출']))
    market2ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장2']['3월매출']))
    market2ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장2']['4월매출']))
    market2ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장2']['5월매출']))
    market2ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장2']['6월매출']))

    market3ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장3']['1월매출']))
    market3ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장3']['2월매출']))
    market3ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장3']['3월매출']))
    market3ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장3']['4월매출']))
    market3ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장3']['5월매출']))
    market3ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장3']['6월매출']))

    market4ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장4']['1월매출']))
    market4ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장4']['2월매출']))
    market4ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장4']['3월매출']))
    market4ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장4']['4월매출']))
    market4ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장4']['5월매출']))
    market4ProfitPerMonth.append(int(store_profit_ex[store_profit_ex['매장'] == '매장4']['6월매출']))


    return render_template('home.html',title='회원현황',max=17000,
                           values=gradeCount,values1=profitPerMarket,
                           m1=market1ProfitPerMonth,m2=market2ProfitPerMonth,m3=market3ProfitPerMonth,m4=market4ProfitPerMonth)
@app.route('/predict')
def predict():
    return render_template('predict.html', result='', interpret = '')

@app.route('/getData', methods=['POST'])
def getData():
    result = ''
    interpret = ''
    stat = request.form['stat']
    gender = request.form['gender']
    married = request.form['married']
    addr = request.form['addr']
    age = request.form['age']
    where = request.form['where']

    age = int(age)
    age_range = ''

    if age >= 70:
        age_range = '70대 이상'
    elif age >= 60:
        age_range = '60대'
    elif age >= 50:
        age_range = '50대'
    elif age >= 40:
        age_range = '40대'
    elif age >= 30:
        age_range = '30대'
    else:
        age_range = '30대 이하'

    dummy_x = [[]]

    dummy_x[0].append(age)

    ## 회원 더미
    if stat == '정상회원':
        dummy_x[0].append(1)
        dummy_x[0].append(0)
    else:
        dummy_x[0].append(0)
        dummy_x[0].append(1)

    ## 성별 더미
    if gender == '남':
        dummy_x[0].append(1)
        dummy_x[0].append(0)
    else:
        dummy_x[0].append(0)
        dummy_x[0].append(1)

    ## 혼인여부 더미
    if married == '기혼':
        dummy_x[0].append(1)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
    elif married == '모름':
        dummy_x[0].append(0)
        dummy_x[0].append(1)
        dummy_x[0].append(0)
    else:
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(1)

    if addr == '경기기타':
        dummy_x[0].append(1)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
    elif addr == '기타':
        dummy_x[0].append(0)
        dummy_x[0].append(1)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
    elif addr == '기흥구':
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(1)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
    elif addr == '동천동':
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(1)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
    elif addr == '분당구':
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(1)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
    elif addr == '상현동':
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(1)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
    elif addr == '서울':
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(1)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
    elif addr == '수원':
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(1)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
    elif addr == '수지구':
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(1)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
    elif addr == '신봉동':
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(1)
        dummy_x[0].append(0)
    else:
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(1)

    ## 나이대 더미
    if age_range == '30대 이하':
        dummy_x[0].append(1)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
    elif age_range == '40대':
        dummy_x[0].append(0)
        dummy_x[0].append(1)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
    elif age_range == '50대':
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(1)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
    elif age_range == '60대':
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(1)
        dummy_x[0].append(0)
    else:
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(1)

    ##주거래  매장 더미
    if where == '매장1':
        dummy_x[0].append(1)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
    elif where == '매장2':
        dummy_x[0].append(0)
        dummy_x[0].append(1)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
    elif where == '매장3':
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(1)
        dummy_x[0].append(0)
    else:
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(0)
        dummy_x[0].append(1)


    predicted = rf.predict(dummy_x)
    predicted = int(predicted[0])

    if predicted == 1:
        result = '유동인구'
        interpret = '서울,경기기타,분당,기흥 주민으로 그날그날 해먹을수 있는 재료 위주로 구입하고 육가공,가루(카레,짜장) 과 같은 즉석식품 구매가 많음'
    elif predicted == 2:
        result = '노인집단(주거래매장:매장3)'
        interpret = '풍덕천,수지구,신봉동 주민으로 육가공,육류 소비가 다른 군집에 비해 작다. 대신에 버섯과 함께 야채위주 소비 하고 표고가루 등 건강에 좋은 가루류 많이 구입 과자 소비가 의외로 많다. 내역을 보면 쌀과자류/전병과자류를 많이 산다'
    elif predicted == 3:
        result = '40-50대 우수고객(주거래매장:매장3)'
        interpret = '풍덕천,수지구,신봉동 주민으로 전체적으로 골고루 많이 산다. 육가공 소비가 두드러 진다'
    else:
        result = '40-50대 일반고객(주거래매장:매장2)'
        interpret = '상현동 주민으로 돼지, 소고기 구매가 두드러 진다.'

    return render_template('predict.html', result=result, interpret = interpret)

@app.route('/maechul')
def maechul():
    return render_template('maechul.html')

if __name__ == '__main__':
    app.run(debug=True)


