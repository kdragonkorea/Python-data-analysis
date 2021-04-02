#### 판다스 기본 구문

- 시리즈(Series)

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
    # iloc/loc
    
    
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
  
    
  
- 파일 입력과 출력

  - 데이터 읽어오기

    ```python
    
    ```

  - 데이터 저장하기

    ```python
    # 데이터프레임
    df.to_csv("./output/df_sample1.csv")
    df.to_csv("./output/df_sample2.csv", index=False)
    ```

- 데이터프레임 구조

  - 미리보기: head, tail

  - 요약정보: shape, info, dtypes, describe

    ```python
    # shape
    # info
    # .dtypes
    # .describe()
    # .describe(include='all')
    ```

  - 열의 데이터 개수

    ```python
    # df.count(): 열의 데이터 개수
    
    # df['열이름'].value_counts(): 열의 고유값 개수
    ```

  - 평균값, 중간값

    ```python
    # df.mean()
    # df.mean()
    ```

  - 최대값, 최소값

    ```python
    # max
    # min
    ```

  - 표준편차, 상관계수

    ```python
    # df.std()
    # df['열 이름'].std()
    
    # df.corr()
    # df['열 이름'].corr()
    ```

- 그래프

  ```python
  # 선 그래프: 
  df.plot()
  
  # 막대 그래프: 
  df.plot(kind='bar')
  	# 행, 열 전치: df1 = df.T
  df.plot(kind='bar', rot=20) # rot는 index명 기울기
  df.plot(kind='bar', color=['a'='red'..]) # color는 리스트 or 딕셔너리로 가능
      
  # 히스토그램: 
  df.plot(kind='hist')
  
  # 산점도: 
  df.plot(kind='scatter')
  df.plot(x='', y='', kind='scatter')
  
  # 박스 플롯: 
  df[['열 이름', ...]].plot(kind='box')
  ```




#### Matplotlib

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




#### Seaborn

- 

#### Folium

- 