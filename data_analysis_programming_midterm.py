import numpy as np
import pandas as pd
import sqlite3
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from sklearn.manifold import TSNE

"""
## Problem 1. Fibonacci [2 points]
Given `n`, implement function `fib(n)` that calculates `n`-th Fibonacci number. <br>
Assume `n` is a positive integer and `fib(0)`=0, `fib(1)`=1.
"""
def fib(n):
    a,b = 0, 1
    for i in range(n):
        a,b = b, a+b
    return a
print(fib(5))

"""
## Problem 2. Maximum Word Count [2 points]
Given `filename`, implement function `max_word_count(filename)` that finds the most frequent words.
You should open a file whose name is `filename` and return two things as a tuple:
The first one is a set of words that occur the maximum number of times and the second one is their counts.
"""
def max_word_count(filename):
    file = open(filename, 'r')
    words = file.split()
    set_of_max_word = set()
    cnt = dict()
    for x in words:
        if x not in cnt: cnt[x] = 0
        else: cnt[x] += 1
    maxval = 0
    for x in cnt: maxval = max(maxval, cnt[x])
    for x in cnt:
        if maxval == cnt[x]:
            set_of_max_word.add(x)
    return set_of_max_word, maxval


"""
## Problem 3. Average Price [3 points]
Given `cars`, implement function `average_prices(cars)` that returns a dictionary which contains each `brand` as a key and the average price of cars of that `brand` as a value.
"""
class car:
    def __init__(self, args):
        brand, model, price = args
        self.brand = brand
        self.model = model
        self.price = price
def average_prices(cars):
    cnt_value = dict()
    cnt_num = dict()
    for x in cars:
        if x.brand not in cnt_value:
            cnt_value[x.brand] = 0
            cnt_num[x.brand] = 0
        cnt_value[x.brand] += x.price
        cnt_num[x.brand] += 1
    avg = dict()
    for x in cnt_value: avg[x] = cnt_value[x] / cnt_num[x]
    return avg


"""
## Problem 4. Manhattan Distance [2 points]
Given two numpy arrays `arr1` and `arr2`, implement function `manhattan_distance` that calculates Manhattan distance between `arr1` and `arr2`. <br>
You need to utilize numpy library for this problem.
"""
def manhattan_distance(arr1, arr2):
    abs_diff = np.abs(arr1 - arr2)
    distance = np.sum(abs_diff)
    return distance

"""
## Problem 5. CSV Modification [5 points]
Your goal is to modify given csv file with below constraints. <br>
The inputs are paths of the original data and modified data. <br>
You need to utilize pandas library for this problem.

### Constraints
- The requirements must be followed in the same order as given below.<br>
  (If not, you might attain different results although you followed everything correctly.)
1. The modified csv file should not contain columns `Cabin` and `Embarked`.
2. The modified csv file should not contain missing values. <br>
   All rows with a missing value needs to be dropped.
3. The modified csv file should only have `survived` = 1.
4. The column `Pclass` should be one hot encoding.
5. Value of `Age` should be `young` if it is smaller than 65 and `old` otherwise.
"""

def titanic(original_file, modified_file):
    df = pd.read_csv(original_file) #csv(comma-seperated value) 읽기
    df = pd.DataFrame(df) #dataframe 형성
    df = df.drop(['Cabin','Embarked'], axis =  'columns') # labels = [classes]가 있는 axis 제거
    df = df.dropna() #NaN값이 포함된 열 제거
    idx = df[df['Survived'] != 1].index #df['Survived'] != 1에 해당하는 data의 index
    df.drop(idx, inplace=True) # label = idx 즉, 해당 index의 row 제거 (inplace = Ture면 copy를 반환하지 않음)
    df = pd.get_dummies(df, columns = ['Pclass'], dtype = int) # one-hot encoding
    df['Age'] = df['Age'].apply(lambda x: 'young' if x < 65 else 'old') # 해당 axis에 해당 함수 수행
    df.to_csv(modified_file, index=False) # csv파일로 변환(row 이름 안쓰게 index = false)

"""
## Problem 6. Employee and Department [6 points]
For this problem, three csv files, `departments.csv`, `employees.csv` and `employees2.csv`, are given. <br>
There are 2 sub problems. <br>
You need to utilize pandas library for this problem.
"""

"""
### 6.a Employee Table [3 points]
Make employee table that has `name`, `salary` and `department_name` as columns. <br>
Note that each department has its own `department_id` and `department_name`.
"""
def emp_table(dep, emp1, emp2):
    dep_table = pd.read_csv(dep) 
    emp1_table = pd.read_csv(emp1)
    emp2_table = pd.read_csv(emp2)
    # 자료 읽기
    emp1 = pd.merge(emp1_table, dep_table, on = 'department_id', how = 'left')
    emp2 = pd.merge(emp2_table, dep_table, on = 'department_id', how = 'left')
    # 두 데이터 프레임을 key를 기준으로 병합
    df = pd.concat([emp1, emp2], ignore_index = True)
    # 두 데이터 프레임을 그냥 병합(index가 넘어올 수 있으므로 무시)
    df = df[["name", "salary", "department_name"]]
    return df

"""
### 6.b Highest Salary [3 points]
Find an employee with the highest salary of each department. <br>
The output must be a dictionary which contains `department_name` as a key and an employee's `name` as a value. <br>
You can use `emp_table` in 6.a.
"""
def highest_salary(dep, emp1, emp2):
    table = emp_table(dep, emp1, emp2)
    df = table.groupby('department_name').max()
    # 해당 값을 기준으로 묶은 뒤에 max값들만 table 형성
    df = df.drop(columns = 'salary') # 필요없는 column 제거
    print(df)
    return df.to_dict()['name']



"""
### a. Create [1 points]
Implement function `create` that connects a database named `titanic.db` and creates a table named `Company` in it. <br>
The contents of table `Company` are as follow:
   | Employee | Department | Salary | Gender |
   | :-       | :-         | :-     | :-     |
   | John     | sales      | 5000   | M      |
   | Allen    | accounting | 6000   | M      |
   | Martin   | research   | 3500   | M      |
   | Mary     | sales      | 5500   | F      |
   | Smith    | research   | 4500   | M      |
"""
def create() -> None:
    '''
        create() connects a database named |titanic.db| and creates a table |Company| in it.

        Columns and data types of table |Company| are as follow:
            |Employee|   - string
            |Department| - string
            |Salary|     - int
            |Gender|     - string
        
        The order of data insertion does not matter.
    '''
    db = sqlite3.connect('titanic.db') # 파일 db 접속
    cursor = db.cursor() # curcor 객체를 받아와 table 생성
    cursor.execute('''CREATE TABLE Company ( Employee TEXT, Department TEXT, Salary INTEGER, Gender TEXT )''')
    dd = [("John", "sales", 5000, "M"),("Allen", "accounting", 6000, "M"),("Martin", "research", 3500, "M"),("Mary", "sales", 5500, "F"),("Smith", "research", 4500, "M")]
    cursor.executemany("INSERT INTO Company VALUES (?, ?, ?, ?)", dd)   
    # list로 데이터 삽입
    db.commit() # 영구 저장
    db.close() # 종료
    return

'''
### b. Select 1 [1 points]
Implement function `select1` that fetches all data from table `Titanic` in database `titanic.db`.
'''
def select1() -> List[Tuple[int, str, int, int]]:
    '''
        select1() fetches all data from table |Titatic| in database |titanic.db|
            and returns them as a list of tuples.

        Columns and data types of table |Titanic| are as follow:
            |Pclass|   - int
            |Name|     - string
            |Survived| - int
            |Age|      - int
        
        The returned list should be formatted as follow:
            [ Tuple_1, ..., Tuple_N ]
            where N is # of fetched data and
            Tuple_n = ( |Pclass|, |Name|, |Survived|, |Age| )
            for 1 <= n <= N.

        The order of tuples does not matter.
    '''
    db = sqlite3.connect('titanic.db')
    cursor = db.cursor()
    query = "SELECT Pclass, Name, Survived, Age FROM Titanic"
    cursor.execute(query)
    # 해당 query에 해당하는 걸로 조회
    rows = cursor.fetchall() # cursor에 들어가 있는 열들 전체 조회
    db.close()
    result = [tuple(row) for row in rows]
    return result

'''
### c. Select 2 [2 points]
Implement function `select2` that fetches all data of survivals under the age of 65 from table `Titanic` in database `titanic.db`. <br>
The data shoulbe be arranged in ascending order of age.
'''
def select2() -> List[Tuple[int, str, int, int]]:
    '''
        select2() fetches all data satisfying certain condition from table |Titatic| in database |titanic.db|
            and returns them as a list of tuples.

        Columns and data types of table |Titanic| are as follow:
            |Pclass|   - int
            |Name|     - string
            |Survived| - int
            |Age|      - int
        
        The returned list should be formatted as follow:
            [ Tuple_1, ..., Tuple_N ]
            where N is # of fetched data and
            Tuple_n = ( |Pclass|, |Name|, |Survived|, |Age| ), |Survived| == 1, |Age| < 65
            for 1 <= n <= N.

        The tuples should be arranged in ascending order of |Age|.
        The order of tuples with the same |Age| does not matter.
    '''
    db = sqlite3.connect('titanic.db')
    cursor = db.cursor()
    query = "SELECT Pclass, Name, Survived, Age FROM Titanic WHERE Survived = 1 AND Age < 65 ORDER BY Age"
    cursor.execute(query)
    rows = cursor.fetchall()
    db.close()
    result = [tuple(row) for row in rows]
    return result

'''
### d. Select 3 [2 points]
Implement function `select3` that fetches all data from table `Titanic` in database `titanic.db` <br>
and calculates average age of passengers by ticket class.
'''
def select3() -> List[Tuple[int, float]]:
    '''
        select3() fetches all data from table |Titatic| in database |titanic.db|,
            calculates average age by |Pclass|, and returns them as a list of tuples.

        Columns and data types of table |Titanic| are as follow:
            |Pclass|   - int
            |Name|     - string
            |Survived| - int
            |Age|      - int
        
        The returned list should be formatted as follow:
            [ Tuple_1, ..., Tuple_N ]
            where N is # of fetched data and
            Tuple_n = ( |Pclass|, average |Age| )
            for 1 <= n <= N.

        The order of tuples does not matter.
    '''
    db = sqlite3.connect('titanic.db')
    cursor = db.cursor()
    query = "SELECT Pclass, AVG(Age) FROM Titanic GROUP BY Pclass"
    cursor.execute(query)
    rows = cursor.fetchall()
    db.close()
    result = [tuple(row) for row in rows]
    return result

'''
### e. Join [2 points]
Implement function `join` that left-joins two tables `Company` and `Titanic` in database `titanic.db` by name. <br>
In this problem, table `Company` is utilized, which means you should successfully implement function `create` to start this problem.
'''
def join() -> List[Tuple[str, str, int, str, int, str, int, int]]:
    '''
        join() left-joins two tables |Company| and |Titatic| in database |titanic.db|
            and returns them as a list of tuples.
        
        Columns and data types of table |Company| are as follow:
            |Employee|   - string
            |Department| - string
            |Salary|     - int
            |Gender|     - string

        Columns and data types of table |Titanic| are as follow:
            |Pclass|   - int
            |Name|     - string
            |Survived| - int
            |Age|      - int
        
        The returned list should be formatted as follow:
            [ Tuple_1, ..., Tuple_N ]
            where N is # of fetched data and
            Tuple_n = ( |Employee|, |Department|, |Salary|, |Gender|, |Pclass|, |Name|, |Survived|, |Age| )
            for 1 <= n <= N.    

        The order of tuples does not matter.
    '''
    db = sqlite3.connect('titanic.db')
    cursor = db.cursor()
    query = '''SELECT A.Employee, A.Department, A.Salary, A.Gender, B.Pclass, B.Name, B.Survived, B.Age FROM Company A LEFT JOIN Titanic B ON A.Employee = B.Name'''
    cursor.execute(query)
    rows = cursor.fetchall()
    db.close()
    result = [tuple(row) for row in rows]
    return result

'''
### a. from SQLite Database to Pandas Dataframe [2 points]
Implement function `db2df` that converts table `Titanic` in database `titanic.db` into pandas dataframe and saves it in `titanic.csv`.
'''
def db2df() -> None:
    '''
        db2df() converts table |Titanic| in database |titanic.db| into pandas dataframe
            and saves it in |titanic.csv|.

        Columns and data types of table |Titanic| are as follow:
            |Pclass|   - int
            |Name|     - string
            |Survived| - int
            |Age|      - int
        
        Columns of the converted dataframe, namely df, should be as follow:
            df.columns == ['Pclass', 'Name', 'Survived', 'Age']

        The order of rows does not matter.
    '''
    db = sqlite3.connect('titanic.db')
    query = "SELECT Pclass, Name, Survived, Age FROM Titanic"
    df = pd.read_sql_query(query, db)
    db.close()
    df.to_csv('titanic.csv', index=False)
    print(df.columns)

'''
### b. from Pandas Dataframe to SQLite Database [2 points]
Implement function `df2db` that converts pandas dataframe read from `titanic2.csv` into table `Titanic2` and saves it in database `titanic.db`.
'''
def df2db() -> None:
    '''
        df2db() converts pandas dataframe read from |titanic2.csv| into table |Titanic2|
            and saves it in |titanic.db|.

        Columns and data types of table |Titanic2| are as follow:
            |Ticket| - string
            |Fare|   - int
        
        The order of columns and data in table |Titanic2| does not matter.
    '''
    df = pd.read_csv('titanic2.csv')
    db = sqlite3.connect('titanic.db')
    df.to_sql('Titanic2', db , if_exists='replace', index=False)
    # to_sql : name, connect, if_exists(이미 테이블이 있을 경우) - 지우고 새로 만듦 : replace, index = false : 인덱스가 안 겹칠 수 있으므로

'''
## Problem 3. Relational Algebra [8 points]
Connect to database `titanic.db`, execute following relational algebra expression, and return fetched data. <br>
In this problem, tables `Company` and `Titanic2` are utilized, which means you should successfully implement functions `create` and `df2db` to start this problem.
'''
'''
### a. [2 points]
$\pi_{Ticket, Fare}(\sigma_{Fare > 30}(Titanic2))$
'''
def rel_alg1() -> List[Tuple[str, int]]:
    '''
        rel_alg1() connects to database |titanic.db|, executes given relational algebra expression,
            and returns fetched data as a list of tuples.
        
        Columns and data types of table |Company| are as follow:
            |Employee|   - string
            |Department| - string
            |Salary|     - int
            |Gender|     - string

        Columns and data types of table |Titanic| are as follow:
            |Pclass|   - int
            |Name|     - string
            |Survived| - int
            |Age|      - int

        Columns and data types of table |Titanic2| are as follow:
            |Ticket| - string
            |Fare|   - int
        
        The returned list should be formatted as follow:
            [ Tuple_1, ..., Tuple_N ]
            where N is # of fetched data and
            Tuple_n = ( |Ticket|, |Fare| )
            for 1 <= n <= N.    

        The order of tuples does not matter.
    '''
    db = sqlite3.connect('titanic.db')
    cursor = db.cursor()
    query = "SELECT Ticket, Fare FROM Titanic2 WHERE Fare > 30"
    cursor.execute(query)
    rows = cursor.fetchall()
    db.close()
    result = [tuple(row) for row in rows]
    return result

'''
### b. [3 points]
$\pi_{Name, Survived}(\sigma_{Pclass = 2}(Titanic)) - \pi_{Name, Survived}(\sigma_{Age \ge 20}(Titanic))$
'''
def rel_alg2() -> List[Tuple[str, int]]:
    '''
        rel_alg2() connects to database |titanic.db|, executes given relational algebra expression,
            and returns fetched data as a list of tuples.
        
        Columns and data types of table |Company| are as follow:
            |Employee|   - string
            |Department| - string
            |Salary|     - int
            |Gender|     - string

        Columns and data types of table |Titanic| are as follow:
            |Pclass|   - int
            |Name|     - string
            |Survived| - int
            |Age|      - int

        Columns and data types of table |Titanic2| are as follow:
            |Ticket| - string
            |Fare|   - int
        
        The returned list should be formatted as follow:
            [ Tuple_1, ..., Tuple_N ]
            where N is # of fetched data and
            Tuple_n = ( |Name|, |Survived| )
            for 1 <= n <= N. 

        The order of tuples does not matter.
    '''
    db = sqlite3.connect('titanic.db')
    cursor = db.cursor()
    query = "SELECT Name, Survived FROM Titanic WHERE Pclass = 2 EXCEPT SELECT Name, Survived FROM Titanic WHERE Age >= 20"
    cursor.execute(query)
    rows = cursor.fetchall()
    db.close()
    result = [tuple(row) for row in rows]
    return result

'''
### c. [3 points]
$\pi_{Employee, Salary, Age}(\sigma_{Employee = Name}(Company \times Titanic))$
'''
def rel_alg3() -> List[Tuple[str, int, int]]:
    '''
        rel_alg3() connects to database |titanic.db|, executes given relational algebra expression,
            and returns fetched data as a list of tuples.
        
        Columns and data types of table |Company| are as follow:
            |Employee|   - string
            |Department| - string
            |Salary|     - int
            |Gender|     - string

        Columns and data types of table |Titanic| are as follow:
            |Pclass|   - int
            |Name|     - string
            |Survived| - int
            |Age|      - int

        Columns and data types of table |Titanic2| are as follow:
            |Ticket| - string
            |Fare|   - int
        
        The returned list should be formatted as follow:
            [ Tuple_1, ..., Tuple_N ]
            where N is # of fetched data and
            Tuple_n = ( |Employee|, |Salary|, |Age| )
            for 1 <= n <= N. 

        The order of tuples does not matter.
    '''
    db = sqlite3.connect('titanic.db')
    cursor = db.cursor()
    query = "SELECT Employee, Salary, Age FROM (SELECT * FROM Company JOIN Titanic ON 1=1) WHERE Employee = Name"
    cursor.execute(query)
    rows = cursor.fetchall()
    db.close()
    result = [tuple(row) for row in rows]
    return result

'''
## Problem 4. Complicated SQL Query [10 points]
In this problem, do not just fetch all data from table and use basic python operations to solve the problem. <br>
Only SQL query is allowed.
'''
'''
### a. Complicated Select 1 [3 ponts]
Implement `comp_select1` that counts the number of passengers for each ticket class using table `Titanic` in database `titanic.db`.
'''
def comp_select1() -> List[Tuple[int, int]]:
    '''
        comp_select1() counts the numbers of passengers for each ticket class using table |Titanic| in database |titanic.db|
            and returns them as a list of tuples.

        Columns and data types of table |Titanic| are as follow:
            |Pclass|   - int
            |Name|     - string
            |Survived| - int
            |Age|      - int
        
        The returned list should be formatted as follow:
            [ Tuple_1, ..., Tuple_N ]
            where N is # of fetched data and
            Tuple_n = ( |Pclass|, # of passengers )
            for 1 <= n <= N. 

        The order of tuples does not matter.
    '''
    db = sqlite3.connect('titanic.db')
    cursor = db.cursor()
    query = "SELECT Pclass, COUNT(*) AS Pcount FROM Titanic GROUP BY Pclass"
    cursor.execute(query)
    rows = cursor.fetchall()
    db.close()
    result = [tuple(row) for row in rows]
    return result
'''
### b. Complicated Select 2 [3 ponts]
Implement `comp_select2` that calculates the average age of passengers for each ticket class using table `Titanic` in database `titanic.db` <br>
and finds ticket classes whose average passenger age is under 35.
'''
def comp_select2() -> List[Tuple[int, int]]:
    '''
        comp_select2() calculates the average passenger age for each ticket class using table |Titanic| in database |titanic.db|,
            finds ticket classes whose average passenger age is under 35,
            and returns them as a list of tuples.

        Columns and data types of table |Titanic| are as follow:
            |Pclass|   - int
            |Name|     - string
            |Survived| - int
            |Age|      - int
        
        The returned list should be formatted as follow:
            [ Tuple_1, ..., Tuple_N ]
            where N is # of fetched data and
            Tuple_n = ( |Pclass|, average passenger age ), average passenger age < 35
            for 1 <= n <= N. 

        The order of tuples does not matter.
    '''
    db = sqlite3.connect('titanic.db')
    cursor = db.cursor()
    query = "SELECT Pclass, AVG(Age) AS Avgage FROM Titanic GROUP BY Pclass Having Avgage < 35"
    cursor.execute(query)
    rows = cursor.fetchall()
    db.close()
    result = [tuple(row) for row in rows]
    return result

'''
### c. Complicated Select 3 [4 ponts]
Implement `comp_select3` that finds every pair of survived passengers using table `Titanic` in database `titanic.db`. <br>
For example, if `A`, `B`, `C` and `D` are all survived passengers, `comp_select3` would return `[(A, B), (A, C), (A, D), (B, C), (B, D), (C, D)]`. <br>
The order of tuples in list can be changed.
'''
def comp_select3() -> List[Tuple[str, str]]:
    '''
        comp_select3() finds every pair of survived passengers' names using table |Titanic| in database |titanic.db|
            and returns them as a list of tuples.

        Columns and data types of table |Titanic| are as follow:
            |Pclass|   - int
            |Name|     - string
            |Survived| - int
            |Age|      - int
        
        The returned list should be formatted as follow:
            [ Tuple_1, ..., Tuple_N ]
            where N is # of fetched data and
            Tuple_n = ( |Name|_1, |Name|_2 ), |Name|_1 and |Name|_2 are lexicographically ordered
            for 1 <= n <= N. 

        The order of tuples does not matter.
    '''
    db = sqlite3.connect('titanic.db')
    cursor = db.cursor()
    query = '''SELECT A.Name, B.Name FROM Titanic A JOIN Titanic B ON A.Survived = 1 AND B.Survived = 1 AND A.Name < B.Name'''
    cursor.execute(query)
    rows = cursor.fetchall()
    db.close()
    result = [tuple(row) for row in rows]
    return result


'''
### a. Basic Plot [3 points]
Draw $y = x^2 - 3$ and $y = 3 - x^2$ graphs on one plot. <br>
Check for detailed properties below:
'''
'''
    Draw y = x^2 - 3 and y = 3 - x^2 graphs on one plot.
    For each graph, add legend describing it at the bottom of the center.

    The properties of the whole figure are as follow:
        |x_label| - 'x'
        |y_label| - 'y'

    The properties of each graph are as follow:

        y = x^2 - 3
            |color|     - red
            |linewidth| - 2
            |linestyle| - dashed
            |legend|    - r'y = x^2 - 3'
        
        y = 3 - x^2
            |color|     - blue
            |linewidth| - 1
            |linestyle| - solid
            |legend|    - r'y = 3 - x^2'
'''

x = np.linspace(-3, 3, 300)
y1 = x**2 - 3
y2 = 3 - x**2
plt.plot(x, y1, label = 'r$y = x^2 - 3$', color = 'red', linewidth = 2, linestyle = 'dashed')
plt.plot(x, y2, label = 'r$y = 3 - x^2$', color = 'blue', linewidth = 1, linestyle = 'solid')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Graphs of $y = x^2 - 3$ and $y = 3 - x^2$')
plt.axhline(0, color = 'black', linewidth = 0.5)
plt.axvline(0, color = 'black', linewidth = 0.5)
plt.show()

'''
### b. Subplot [3 points]
Draw $y = sin(x)$, $y = cos(x)$, and $y = tan(x)$ graphs on three subplots with one row and three columns. <br>
Draw $y = sin(x)$ on the left, $y = cos(x)$ on the middle, and $y = tan(x)$ on the right. <br>
Check for detailed properties below:
'''
'''
    Draw y = sin(x), y = cos(x), and y = tan(x) graphs on three subplots with one row and three columns.
    On the left side, draw y = sin(x) as a normal plot.
    On the middle side, draw y = cos(x) as a scatter plot.
    On the right side, draw y = tan(x) as a normal plot.

    The properties of the whole figure are as follow:
        |size|  - (10, 5)
        |title| - 'Trigonometric functions'

    The properties of each graph are as follow:

        y = sin(x)
            |title|   - 'y = sin(x)'
            |color|   - red
        
        y = cos(x)
            |title|   - 'y = cos(x)'
            |color|   - green
        
        y = tan(x)
            |title|   - 'y = tan(x)'
            |color|   - blue
            |y_range| - [-3, 3]
'''

x = np.linspace(-np.pi, np.pi, 50)

y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

figure, axes = plt.subplots(1, 3, figsize = (10, 5))
figure.suptitle('Trigonometric functions', fontsize = 15)

axes[0].plot(x, y1, color = 'red')
axes[0].set_title('y = sin(x)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

axes[1].scatter(x, y2, color = 'green')
axes[1].set_title('y = cos(x)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

axes[2].plot(x, y3, color = 'blue')
axes[2].set_title('y = tan(x)')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].set_ylim(-3, 3)

'''
### c. Histogram [4 points]
Draw $\mathcal{N}(0, 1)$ graph and histogram of normally sampled values on one plot. <br>
Check for detailed properties below:
'''
'''
    Draw N(0, 1) graph and histogram of normally sampled values on one plot.

    The properties of the whole figure are as follow:
        |title| - 'Theoretical and Statistical graph for Normal Distribution'

    The properties of each graph are as follow:
        
        N(0, 1) graph
            |color|     - red
        
        histogram
            # of |bin|  - 25 (each bin should have the same width)
            |color|     - blue
            |alpha|     - 0.7 (for mode bin) / 0.5 (otherwise)
'''

x = np.linspace(-4, 4, 100)

rv = scipy.stats.norm()
theo = rv.pdf(x)

num_sample = 10000
stat = np.random.normal(0, 1, num_sample)

figure, axes = plt.subplots(figsize = (10, 6))
figure.suptitle('Theoretical and Statistical graph for Normal Distribution', fontsize = 15)

axes.plot(x, theo, color = 'red', label = '${N}(0, 1)$')

n, histo, patches = axes.hist(stat, bins = 25, density = True, color = 'blue', alpha = 0.5, label = 'Histogram')
# data, bins = 가로축 구간의 개수, density = return pdf
mode_bin = np.argmax(n) # 가장 큰 값 찾기
patches[mode_bin].set_alpha(0.7)

axes.legend()
plt.show()

'''
### d. Qunatile Plot [2 points]
Draw 20-quantile (i.e., vigiciles) plot of $U(-3, 3)$. <br>
Check for detailed properties below:
'''
'''
    Draw 20-quantile (i.e., vigiciles) plot of U(-3, 3).

    The properties of each graph are as follow:
        
        quantile plot
            quantile - 20
            |color|  - red
            |marker| - point
'''

a = np.random.uniform(-3, 3, size=1000)
q = np.linspace(0, 100, 21)

quan = np.percentile(a, q)
plt.figure(figsize = (8, 6))
plt.plot(q, quan, marker = '.', color  = 'red')
plt.title('20-quantile Plot of U(-3, 3)', fontsize = 15)
plt.grid(True)
plt.show()

'''
### e. Q-Q Plot [3 points]
Draw Q-Q plot comparing $U(-3, 3)$ and $\mathcal{N}(0, 1)$. <br>
Use 20-quantile (i.e., vigiciles) for both distributions. <br>
Also draw a guiding line, $y = x$ graph. <br>
Check for detailed properties below:
'''
'''
    Draw Q-Q plot comparing U(-3, 3) and N(0, 1).
    Use 20-quantile (i.e., vigiciles) for both distributions.
    Also draw a guiding line, y = x graph.

    The properties of the whole figure are as follow:
        |x_label| - 'Uniform(-3, 3)'
        |y_label| - 'Normal(0, 1)'

    The properties of each graph are as follow:
        
        Q-Q plot
            quantile - 20 (for both)
            |color|     - blue
            |linestyle| - no line (use linestyle='')
            |marker|    - point
            
        
        y = x
            |color|     - black
            |linestyle| - dashed
            
'''

a = np.random.uniform(-3, 3, size=1000)
b = np.random.normal(0, 1, size=1000)
q = np.linspace(0, 100, 21)
x = np.linspace(-3, 3, 100)

p = np.percentile(a, q)
r = np.percentile(b, q)
plt.figure(figsize = (8, 6))
plt.plot(p, r, marker = '.', color = 'blue', linestyle = ' ', label = 'Q-Q Plot')
plt.plot([-3, 3], [-3, 3], color = 'black', linestyle = 'dashed', label = 'y = x')
plt.title('Q-Q Plot: Uniform(-3, 3) vs. Normal(0, 1)', fontsize = 15)
plt.xlabel('Uniform(-3, 3)')
plt.ylabel('Normal(0, 1)')
plt.legend()
plt.grid(True)
plt.show()

'''
## Problem 2. Seaborn [6 points]
For this problem, only `seaborn` is allowed to be used. <br>
Do not use `matplotlib` or any other kind of visualization library.
'''
'''
    Data types and descriptions of columns in dataset |titanic| are as follow:
        |alive|    -> str : Survival
        |class|    -> str : Ticket class
        |sex|      -> str : Sex (male, female)
        |age|      -> int : Age in years (0 - 80)
        |fare|     -> int : Passenger fare (0 - 512)
        |embarked| -> str : Port of Embarkation (C: Cherbourg, Q: Queenstown, S: Southampton)
'''

columns = ['alive', 'class', 'sex', 'age', 'fare', 'embarked']
data = sns.load_dataset('titanic').dropna().reset_index(drop=True)[columns]
data['age'] = data['age'].apply(int)
data['fare'] = data['fare'].apply(int)
data
'''
### a. Box plot [1 points]
Draw a box plot comparing the `age` distribution of each `alive`.
'''
'''
    Draw a box plot comparing the age distribution of each alive.
'''
sns.boxplot(x = 'alive', y = 'age', data = data)
'''
### b. Bar plot [1 points]
Draw a bar plot comparing the `age` distribution of each `class`. <br>
For each `class`, show `age` distribution for each `alive`.
'''
'''
    Draw a bar plot comparing the fare distribution of each class.
    For each class, show age distribution for each alive.
'''
sns.barplot(x = 'class', y = 'age', hue = 'alive', data = data)
'''
### c. Histogram and KDE (Kernel Density Estimation) [1 points]
Draw a histogram and KDE for `fare`.
'''
'''
    Draw a histogram and KDE for fare
'''
sns.histplot(data = data, kde = True, x = 'fare')
'''
### d. Joint plot [1 points]
Draw a joint plot between `age` and `fare`.
'''
'''
    Draw a joint plot between age and fare.
'''
sns.jointplot(data = data, x = 'age', y = 'fare')
'''
### e. Heatmap [2 points]
Draw a heatmap between `class` and `embarked` where values are `age`.
'''
'''
    Draw a heatmap between class and embarked where values are age.
'''
pivot_table = data.pivot_table(index = 'class', columns = 'embarked', values = 'age')
sns.heatmap(pivot_table)

'''
### Load Dataset [0 points]
`iris` dataset is used for this problem.
'''
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
X, y = data['data'], data['target']
X, y
'''
### a. Dimensionality Reduction [2 points]
Reduce dimension of each data in `X` into 2. <br>
Set column names of transformed `X`, `X_tnse`, as `component_0` and `component_1`.
'''
'''
    Reduce dimension of each data in |X| into 2.
    Set column names of transformed |X|, |X_tnse|, as |component_0| and |component_1|.
    Note that |X_tnse| should still be pandas dataframe.
'''
tsne = TSNE(n_components = 2)
X_tsne_np = tsne.fit_transform(X) # 훈련 후 적용
X_tsne = pd.DataFrame(X_tsne_np, columns = ['component_0', 'component_1'])

'''
### b. Visualization [2 points]
Visualize `X_tnse` for each `y` on one plot using `matplotlib`. <br>
Check for detailed properties below:
'''
'''
    Visualize |X_tnse| for each |y| on one plot.
    That is, draw one scatter plot for each |y| value.
    For each graph, add legend describing it.

    The properties of the whole figure are as follow:
        |title|   - 't-SNE result for Iris dataset'
        |x_label| - 'component_0'
        |y_label| - 'component_1'

    The properties of each graph are as follow:

        y = 0
            |color|  - 'red'
            |legend| - 'class_0'

        y = 1
            |color|  - 'green'
            |legend| - 'class_1'

        y = 2
            |color|  - 'blue'
            |legend| - 'class_2'
'''

# BEGIN_YOUR_CODE
X_tsne['target'] = data['target']

df0 = X_tsne[X_tsne['target'] == 0]
df1 = X_tsne[X_tsne['target'] == 1]
df2 = X_tsne[X_tsne['target'] == 2]

plt.scatter(df0['component_0'], df0['component_1'], color = 'red', label = 'class_0')
plt.scatter(df1['component_0'], df1['component_1'], color = 'green', label = 'class_1')
plt.scatter(df2['component_0'], df2['component_1'], color = 'blue', label = 'class_2')

plt.title('t_SNE result for Iris dataset')
plt.xlabel('component_0')
plt.ylabel('component_1')
plt.legend()
plt.show()