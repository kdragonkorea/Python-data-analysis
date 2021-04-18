[TOC]

# 판다스 자료 구조

### 시리즈(Series)와 데이터프레임(df)

- 시리즈 만들기

  ```python
  import pandas as pd
  
  # 딕셔너리
  dict_data = {'a': 1, 'b': 2, 'c': 3}
  sr = pd.Series(dict_data)
  
  # 리스트
  list_data = ['2019-01-02', 3.14, 'ABC', 100, True]
  sr = pd.Series(list_data)
  
  # range
  sr = pd.Series(range(10,20,2))
  
  # 튜플
  tup_data = ('영인', '2010-05-01', '여', True)
  sr = pd.Series(tup_data, index=['이름', '생년월일', '성별', '학생여부'])
  ```

- 시리즈 인덱스(index)와 벨류(values)

  ```python
  idx = sr.index
  val = sr.values
  ```

- 시리즈 원소 선택

  ```python
  # 원소를 1개 선택
  print(sr[0])       # sr의 1 번째 원소를 선택 (정수형 위치 인덱스를 활용)
  print(sr['이름'])  # '이름' 라벨을 가진 원소를 선택 (인덱스 이름을 활용)
  
  # 여러 개의 원소를 선택 (인덱스 리스트 활용)
  print(sr[[1, 2]]) 
  print(sr[['생년월일', '성별']])
  
  # 여러 개의 원소를 선택 (인덱스 범위 지정)
  print(sr[1 : 2])  # 숫자를 지정할 때는 +1을 고려해서 입력해야한다.
  print(sr['생년월일' : '성별'])
  ```

- 데이터프레임
  
  - 데이터프레임 만들기
  
    ```python
    import pandas as pd
    
    # 딕셔너리 (열이름을 key, 리스트를 value)
    dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], 'c3':[10,11,12], 
                 'c4':[13,14,15]}
    df = pd.DataFrame(dict_data)
    
    # 리스트 (행 인덱스/열 이름 지정)
    df = pd.DataFrame([[15, '남', '덕영중'], [17, '여', '수리중']], 
                       index=['준서', '예은'],
                       columns=['나이', '성별', '학교'])
    
    # 리스트2 (행 인덱스/열 이름 지정)
    data = (['둘리', 90, 99, 90], ['또치',80, 98, 70], ['도우너', 70, 97, 70], 
            ['희동이', 70, 46, 60])
    index = ['학생1', '학생2', '학생3', '학생4']
    columns = ['name', 'kor', 'eng', 'mat']
    df = pd.DataFrame(data, index, columns)
    ```
  
  - 행,열 이름 설정/변경
  
    ```python
    # 행과 열이름 설정
    df.index=['n1', 'n2' ...]
    df.columns=['n1', 'n2', 'n3' ...]
    
    # 열 이름 변경
    df.rename(columns={'현재 열 이름':'변경할 열 이름', ...}, inplace=True)
    
    # 행 이름 변경
    df.rename(index={'현재 행 이름':'변경할 행 이름', ...}, inplace=True)
    ```
  
  - 행,열 삭제
  
    ```python
    # axix=0 은 행 삭제, axix=1 열 삭제
    df2 = df.copy()
    df2.drop(['행 또는 열 이름' ..], axis=0, inplace=True)
    ```
  
  - 행,열 선택
  
    ```python
    # 'iloc'는 반드시 숫자만, 'loc'는 이름만 되지만 숫자도 이름처럼 사용 가능
    # 행 정보만 출력
    df.iloc[행 번호]
    df.iloc[행 번호:행 번호-1] # ex df.iloc[2:4] (2행, 3행 선택)
    
    # 열 정보만 출력
    df.
    
    # 행,열 정보 출력
    df.iloc[행 번호, [열 번호]] # ex df.iloc[2:4, [2,3]] (2행, 3행 선택)
    
    # 원소 출력
    df.iloc[행 번호]
    
    ```
  
  - 원소 선택
  
    ```python
    # 
    ```
  
  - 행,열 추가
  
    ```
    
    ```
  
  - 원소 값 변경
  
  - 행,열 위치 변경
  
  - 행 인덱스 재배열
  
  - 행 인덱스 정렬, 열 기준 정렬
  
    ```python
    # 열 기준 정렬 (오름차순)
    df.sort_values('열 이름')
    
    # 열 기준 정렬 (내림차순)
    df.sort_values('열 이름', ascending=False)
    df.sort_values(by = '열 이름', ascending=False)
    ```





___



### 파일 입력과 출력

- 데이터 읽어오기

  ```python
  import pandas as pd
  
  # 1) csv 파일 불러오기
  pd.read_csv('./파일명')
  pd.read_csv('./파일명', header=None)
  pd.read_csv('./파일명', index_col=None)
  df4 = pd.read_csv(file_path, index_col='c0')
  print(df4)
  >
      c1  c2  c3
  c0            
  0    1   4   7
  1    2   5   8
  2    3   6   9
  
  # 2) 컬럼의 최대 길이 정보 확인
  pd.get_option('display.max_columns')
  
  # 3) 컬럼의 최대 길이 설정
  pd.set_option('display.max_columns', 최대길이)
  
  # 4) 설정 옵션 리셋
  pd.reset_option("^display")
  
  # 5) excel파일(xml) 불러오기
  pd.read_excel('./파일명')
  
  # 6) json파일 불러오기
  pd.read_json('./파일명')
  
  # 7) html의 table 태그를 불러오기
  pd.read_html('./파일명')
  
  # 8) index 변경/설정
  df.set_index(['컬럼명'], inplace=True)
  ```

- 데이터 저장하기

  ```python
  import pandas as pd
  
  # 데이터프레임
  df.to_csv("./output/df_sample1.csv")
  df.to_csv("./output/df_sample2.csv", index=False)
  
  # 데이터프레임을 json 형식으로 저장
  df.to_json("./output/df_sample.json")
  
  # 데이터프레임을 excel 형식으로 저장
  df.to_excel("./output/df_sample.xlsx")
  
  # 데이터프레임을 excel 형식의 sheet별로 저장
  writer = pd.ExcelWriter("./output/df_excelwriter.xlsx")
  df1.to_excel(writer, sheet_name="sheet1")
  df2.to_excel(writer, sheet_name="sheet2")
  writer.save()
  ```

___



### 데이터프레임 구조

- 미리보기: head, tail

  ```python
  pd.head() # 상위 5개만 출력 (디폴트: 5개)
  pd.tail() # 하위 5개만 출력 (디폴트: 5개)
  ```

- 요약정보: shape, info, dtypes, describe

  ```python
  # shape: 행과 열의 개수만 출력 (tuple)
  df.shape()
  > (398, 9)
  
  # info: columns의 type, null 정보 출력
  df.info()
  
  # dtypes: 모든 columns의 type 정보 출력 (series)
  df.dtypes
  
  # df.열이름(c1).dtypes: c1의 컬럼 type 출력
  df.c1.dtypes
  
  # describe(): columns의 count, mean, max, min 등 출력
  df.describe() # 일부 컬럼만 출력
  df.describe(include='all') # 모든 컬럼 출력
  
  ```

- 열의 데이터 개수

  ```python
  # df.count(): 열의 데이터 개수
  # df['열이름'].value_counts(): 열의 고유값 개수
  df['col'].value_counts() # col열의 데이터 개수 출력
  ```

- 평균값, 중간값

  ```python
  # df.mean()
  df.col.mean() # col열의 평균값 출력
  df['col'].mean() # col열의 평균값 출력
  
  # df.median()
  ```

- 최대값, 최소값

  ```python
  # df.max()
  # df.min()
  ```

- 표준편차, 상관계수

  ```python
  # df.std(): columns의 표준편차 출력
  # df['열 이름'].std()
  
  # df.corr(): columns의 상관계수 출력
  df[['mpg','weight']].corr()
  >             mpg    weight
  mpg     1.000000 -0.831741
  weight -0.831741  1.000000
  ```



___



### 판다스 내장 시각화 함수(그래프)

```python
# 선 그래프: 
df.plot()

# 막대 그래프: 
df.plot(kind='bar') #세로 막대 그래프
sr.plot(kind='barh') #가로 막대 그래프

# 막대 그래프2: 
# 행, 열 전치: df1 = df.T
df.plot(kind='bar', rot=20) # rot는 index명 기울기
df.plot(kind='bar', color=['a'='red'..]) # color는 리스트 or 딕셔너리로 가능
df.plot(kind='bar', stacked=True) # 막대 하나에 컬럼을 쌓아서 출력
    
# 히스토그램: 
df.plot(kind='hist')

# 산점도: 
df.plot(kind='scatter')
df.plot(x='컬럼명1', y='컬럼명2', kind='scatter')

# 박스 플롯: 
df[['열 이름', ...]].plot(kind='box')

# pie(원형) 그래프
df.plot(kind='pie', subplots=True) # 모든 컬럼의 원형 그래프 출력
df.plot(kind='pie', y='컬럼명') # 특정 컬럼의 원형 그래프 출력
```



___



# Matplotlib

- plot

  ```python
  from matplotlib import pyplot as plt
  plt.figure(figsize=(10,6))
  plt.plot([1,2,3,4,5,6])
  plt.title('matplotliab 그래프(1)')
  plt.xlabel('X-축')
  plt.ylabel('Y-축')
  plt.show()
  ```



# Seaborn

- heatmap

  ```
  sns.heatmap(df, annot=True)
  sns.heatmap(df, annot=True, cmap="PuRd")
  sns.heatmap(df, annot=True, fmt='.1f')
  ```

- barplot

  ```python
  
  ```

- countplot

  ```python
  # 예제 4-31
  sns.countplot(x='class', palette='Set1', data=titanic, ax=ax1)
  ```

- boxplot / violinplot

  ```python
  # 예제 4-32
  sns.boxplot(x='alive', y='age', data=titanic, ax=ax1) 
  sns.violinplot(x='alive', y='age', data=titanic, ax=ax3)
  ```

- jointplot

  ```python
  # 예제 4-33
  j1 = sns.jointplot(x='fare', y='age', data=titanic) 
  ```

- scatterplot

- lineplot

- relplot

- stripplot

- swarmplot

- pointplot

.. exam7 참조



# Folium

- 





# 데이터 전처리(Preprocessing)

- 누락 데이터

  - 확인

    ```python
    # 열의 NaN 개수 계산하기
    df['열 이름'].value_counts(dropna=False)
    
    # isnull() 메서드로 누락 데이터 찾기
    df.isnull()
    
    # notnull() 메서드로 누락 데이터 찾기
    df.notnull()
    
    # isnull() 메서드로 누락 데이터 개수 구하기
    df.isnull().sum(axis=0)
    ```

  - 제거

    ```python
    # NaN 값이 500개 이상인 열을 모두 삭제
    df.dropna(axis=1, thresh=500)
    
    # age 열에 나이 데이터가 없는 모든 행 삭제
    df.dropna(subset=['age'], how='any', axis=0)
    ```

  - 치환

    ```python
    # age 열의 Nan 값을 다른 나이 데이터의 평균으로 변경
    mean_age = df['age'].mean(axis=0)
    df['age'].fillna(mean_age, inplace=True)
    
    # embark_town 열의 NaN 값을 승선도시 중에서 가장 많이 출현한 값으로 치환
    most_freq = df['embark_town'].value_counts(dropna=True).idxma()
    df['embark_town'].fillna(most_freq, inplace=True)
    
    # 서로 이웃하고 있는 데이터끼리 치환
    df['embark_town'].fillna(method='ffill', inplace=True)
    ```

    

- 중복 데이터

  - 확인
  - 제거

- 단위 환산

- 자료형 변환

  ```
  
  ```

  

- 구간 분할

- 더미 변수

- 정규화

- 시계열 데이터

  - 문자열을 Timestamp로 변환

    ```
    pd.to_datetime(df[])
    ```

    

  - Timestamp를 Period로 변환

  - Timestamp 배열

    ```python
    # Timestamp의 배열 만들기 - 월 간격, 월의 시작일 기준
    pd.date_range(start='2019-01-01',  # 날짜 범위 시작
                  end=None,            # 날짜 범위 끝
                  periods=6,           # 생성할 TIMESTAMP 개수
                  freq="MS",           # 시간 간격 (MS: 월의 시작일)
                  tz='Asia/Seoul')     # timezone(시간대)
    
    ```

    * freq 옵션의 종류

      | 옵션 |     설명     | 옵션 |             설명              |
      | :--: | :----------: | :--: | :---------------------------: |
      |  D   |   day(1일)   |  B   |    business day(휴일 제외)    |
      |  W   |     week     |  H   |          hour(1시간)          |
      |  M   |  month end   |  T   |          minute(1분)          |
      |  MS  | month begin  |  S   |          second(1초)          |
      |  Q   |  quater end  |  L   |     milisecond(1/1,000초)     |
      |  QS  | quater begin |  U   |  microsecond(1/1,000,000초)   |
      |  A   |   year end   |  N   | nanosecond(1/1,000,000,000초) |
      |  AS  |  year begin  | ...  |              ...              |

  - Period 배열

  - 날짜 데이터 분리

    ```python
    # dt.year, dt.month, dt.day 방법
    pd.to_datetime(df['Date']) # df에 새로운 열로 추가
    df['new_Date'].dt.year
    df['new_Date'].dt.month
    df['new_Date'].dt.day
    
    # dt.to_period(freq="A"), dt.to_period(freq="M") 방법
    ```

  - 날짜 인덱스 활용



### 데이터프레임의 다양한 응용

- 함수 매핑

  - 개별 원소에 함수 매핑

  - 시리즈 객체에 함수 매핑

  - 데이터프레임 객체에 함수 매핑

    ```
    pipe
    ```

    

- 열 재구성

  - 열 순서 변경
  - 열 분리

- 필터링

  - 불린 인덱싱
  - isin() 메소드 활용

- 데이터프레임 합치기

  - 데이터프레임 연결 (concat)
  - 데이터프레임 병합 (merge)
  - 데이터프레임 결합 (join)

- 그룹 연산

  - 그룹 객체 만들기

    - 1개 열을 기준으로 그룹화

      ```python
      # df.groupby(기준이 되는 열)
      grouped_two = df.groupby(['class'])
      
      # get_group
      df.groupby('city').get_group('부산')
      
      # df.groupby.size()
      df.groupby('city').size()['부산']
      
      # df.groupby.count()
      df.groupby('city').count()
      ```

      

    - 여러 열을 기준으로 그룹화

      ```python
      # df.groupby(기준이 되는 열의 리스트)
      grouped_two = df.groupby(['class', 'sex'])
      
      # get_group
      df.groupby(['city', 'fruits']).get_group(('부산', 'orange'))
      ```

      

  - 그룹 연산 메소드

    - 데이터 집계 (agg)

      ```python
      # group객체.agg(매핑 함수)
      # group객체.agg([함수1, 함수2, 함수3, ...])
      # group객체.agg({'열1':함수1, '열2':함수2, ...})
      ```

      

    - 그룹 연산 데이터 변환 (transform)

      ```python
      # group객체.transform(매핑 함수)
      
      ```

      

    - 그룹 객체 필터링

      ```python
      # group객체.filter(조건식 함수)
      
      # 데이터 개수가 200개 이상인 그룹만을 필터링하여 데이터프레임으로 반환
      grouped_filter = grouped.filter(lambda x: len(x) >= 200)  
      
      # age 열의 평균이 30보다 작은 그룹만을 필터링하여 데이터프레임으로 반환
      age_filter = grouped.filter(lambda x: x.age.mean() < 30)  
      ```

      

    - 그룹 객체에 함수 매핑 

      ```python
      # group객체.apply(조건식 함수)
      
      # 집계 : 각 그룹별 요약 통계정보를 집계
      agg_grouped = grouped.apply(lambda x: x.describe())   
      ```

      

- 멀티 인덱스

  - loc 인덱서

    ```python
    
    ```

    

  - xs 인덱서

    ```python
    # xs 인덱서 사용 - 행 선택(default: axis=0)
    print(pdf3.xs('First'))              # 행 인덱스가 First인 행을 선택 
    print('\n')
    print(pdf3.xs(('First', 'female')))   # 행 인덱스가 ('First', 'female')인 행을 선택
    print('\n')
    print(pdf3.xs('male', level='sex'))  # 행 인덱스의 sex 레벨이 male인 행을 선택
    print('\n')
    print(pdf3.xs(('Second', 'male'), level=[0, 'sex']))  # Second, male인 행을 선택
    print('\n')
    
    # xs 인덱서 사용 - 열 선택(axis=1 설정)
    print(pdf3.xs('mean', axis=1))        # 열 인덱스가 mean인 데이터를 선택 
    print('\n')
    print(pdf3.xs(('mean', 'age'), axis=1))   # 열 인덱스가 ('mean', 'age')인 데이터 선택
    print('\n')
    print(pdf3.xs(1, level='survived', axis=1))  # survived 레벨이 1인 데이터 선택
    print('\n')
    print(pdf3.xs(('max', 'fare', 0), 
                  level=[0, 1, 2], axis=1))  # max, fare, survived=0인 데이터 선택
    ```

    

- 피벗

  ```python
  pdf2 = pd.pivot_table(df,                       # 피벗할 데이터프레임
                       index='class',             # 행 위치에 들어갈 열
                       columns='sex',             # 열 위치에 들어갈 열
                       values='survived',         # 데이터로 사용할 열
                       aggfunc=['mean', 'sum'])   # 데이터 집계 함수
  
  print(pdf2.head())
  ```



# Numpy

- 정리중



# 정규표현식 (import.re)

- re.sub()

  ```python
  import re
  word = "JAVA   가나다 javascript Aa 가나다 AAaAaA123 %^&* 파이썬"
  print(re.sub("A", "", word)) # A만 제거
  print(re.sub("a", "", word)) # a만 제거
  print(re.sub("Aa", "", word)) # 'Aa'만 제거
  print(re.sub("(Aa){2}", "", word)) # 'AaAa' 만 제거
  print(re.sub("[Aa]", "", word)) #'A와 a' 모두 제거
  print(re.sub("[가-힣]", "", word)) # 한글만 제거
  print(re.sub("[^가-힣]", "", word)) # 한글이 아닌 것만 제거 (공백도 제거)
  print(re.sub("[&^%*]", "", word)) # '&^%*' 모두 제거
  print(re.sub("[^가-힣A-Za-z0-9\s]", "", word)) # 한글, 영문, 숫자, 공백이 아닌 것 제거 (특수문자 제거)
  print(re.sub("[\w\s]", "", word)) # 문자(한글,영문,숫자)를 모두 제거
  print(re.sub("\s", "", word)) # 공백을 모두 제거
  print(re.sub("\d", "", word)) # 숫자만 제거
  print(re.sub("\D", "", word)) # 숫자가 아닌 것 제거
  print(re.sub("[^\w]", "", word)) # 문자(한글,영문,숫자)가 아닌 것을 제거
  print(re.sub("\W", "", word)) # 문자가 아닌 것을 제거
  print(re.sub("[^가-힣\s]", "", word)) # 한글과 공백이 아닌 것만 제거
  
  new_word = re.sub("\s+", " ", new_word) # 공백이 1개 이상을 공백 1개로 변경
  print(new_word)
  print(new_word.strip()) # 한글과 공백만 남고 문장 처음에 공백 제거
  ```

- split(), join()

  ```python
  # split
  site = 'web-is-free'
  print(site.split('-'))
  
  # join
  site = [ "web", "is", "free" ]
  print("-".join(site))
  ```



# 텍스트 분석 및 시각화

### 형태소 분석기 (konlpy)

|
|
|

|       Okt        | Komoran | kkma | Hannanum |
| :--------------: | :-----: | :--: | :------: |
| 가장 많이 사용함 |    -    |  -   |    -     |

- kkma

  ```python
  from konlpy.tag import Kkma
  
  sample = '이것은 형태소 분석기 입니다 아버지가방에들어가신다'
  kkma = Kkma() 
  pprint(kkma.nouns(sample))     # 명사 추출
  pprint(kkma.morphs(sample))    # 명사,조사 추출
  pprint(kkma.sentences(sample)) # 문장 추출
  pprint(kkma.pos(sample))       #  추출
  ```

- hannanum

  ```python
  from konlpy.tag import Hannanum  
  
  sample = '이것은 형태소 분석기 입니다 아버지가방에들어가신다'
  hannanum = Hannanum() 
  pprint(hannanum.nouns(sample))
  pprint(hannanum.morphs(sample))
  pprint(hannanum.pos(sample))
  ```

- okt

  ```python
  from konlpy.tag import Kkma
  
  sample = '이것은 형태소 분석기 입니다 아버지가방에들어가신다'
  kkma = Kkma() 
  pprint(kkma.nouns(sample))
  pprint(kkma.morphs(sample))
  pprint(kkma.pos(sample))
  ```

- komoran

  ```python
  from konlpy.tag import Komoran
  
  sample = '이것은 형태소 분석기 입니다 아버지가방에들어가신다'
  komoran = Komoran()
  pprint(komoran.nouns(sample))
  pprint(komoran.morphs(sample))
  pprint(komoran.pos(sample))
  ```




### 워드클라우드

- 텍스트

  ```python
  from wordcloud import WordCloud  ## 워드 클라우드 모듈
  
  text = "둘리 도우너 또치 마이콜 희동이 둘리 둘리 도우너 또치 토토로 둘리 올라프 토토로 올라프 올라프"
  
  wc = WordCloud(
      font_path = myfontpath,
      background_color='white',             ## 배경색
      width = 800,
      height = 800
  )
  wc = wc.generate(text)
  fig = plt.figure()
  plt.imshow(wc, interpolation='bilinear')  ##
  plt.axis('off')
  plt.show()
  
  # png저장
  wc.to_file('output/wordcloud.png')
  ```

- 딕셔너리

  ```python
  keywords = {'파이썬':7, '넘파이':3, '판다스':5, '매트플롭립':2, '시본':2, '폴리엄':2}  
  ## 특정 단어의 빈도를 딕셔너리로 만든다 
  
  wc = wc.generate_from_frequencies(keywords)        ## 빈도별로 워드클라우드를 만든다 
  
  fig = plt.figure()
  plt.imshow(wc, interpolation='bilinear')
  plt.axis('off')
  plt.show()
  ```

- 이미지 형태의 워드클라우드

  ```python
  from PIL import Image
  r2d2_mask = np.array(Image.open('data/r2d2.JPG'))  ## 이미지를 읽어와서 다차원 배열로 변환
  
  wc = WordCloud( stopwords=stopwords,              ## 워드 클라우드 객체를 만든다 
                            font_path = myfontpath,
                            background_color='white',
                             width = 800,
                             height = 800,
                            mask=r2d2_mask)            ## 마스크 인자에 이미지를 전달한다 
  
  texts = ['로봇 처럼 표시하는 것을 보기 위해 이것 은 예문 입니다 가을이라 겨울 바람 솔솔 불어오니 ',
           '여러분 의 문장을 넣 으세요 ㅎㅎㅎ 스타워즈 영화에 나오는 다양한 로봇처럼 r2d2']
  
  ## 두 개의 문자을 연결해서 워드클라우드를 만든다 
  wc = wc.generate_from_text(texts[0]+texts[1])
  wc
  
  ## 이미지를 출력하면 전달된 모양에 따라 표시한다 
  plt.figure(figsize=(8,8))
  plt.imshow(wc, interpolation="bilinear")         
  plt.axis("off")
  plt.show()
  ```




### 자연어 처리 (nltk)

> nltk: Natural Language Toolkit





# 탐색적 데이터 분석 (EDA)

일반적으로 데이터 분석의 절차는 `EDA -> 전처리 -> 분석` 



