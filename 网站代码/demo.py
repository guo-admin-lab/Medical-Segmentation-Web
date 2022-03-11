import sqlite3

conn = sqlite3.connect('db.sqlite3')

# print("Yes")

cursor = conn.cursor()

# cursor.execute("select name from sqlite_master where type='table' order by name")
# print(cursor.fetchall())

# cursor.execute('''select * from Feedback ''')
# print(cursor.fetchall())

cursor.execute('''drop table PersonalInfo''')