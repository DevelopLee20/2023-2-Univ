import oracledb
import pandas as pd

con = oracledb.connect(user="Term", password="123123", dsn="127.0.0.1:1521/SID")
print(con)
