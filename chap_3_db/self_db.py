
import sqlite3
class Database:
    # self._conn =sqlite3.connect('./test.db')는 사용 위치에 주의해야한다.
    def __init__(self):
        self._conn =sqlite3.connect('sqlit.db')  # 공유 되는 지점이다.

    # 아래 패턴이 그냥 써줌
    def create(self):  # table
        sql = """
            CREATE TABLE IF NOT EXISTS Persons ( #IF NOT EXISTS추가됨
                personid varchar(10) primarykey;
                password varchar(10)
                name varchar(10),
                phone varchar(15),
                address varchar(10),
                regdate date defaul:current_timetamp
);
        """
        print('쿼리체크:{}'.format(sql))
        self._conn.execute(sql)
        self._conn.commit()  # self._conn.execute(sql)  self._conn.commit()같이 다님

    def insert_one(self, userid, password, name, phone, address):
        sql = """
                   INSERT INTO Persons(
                   personid, password, name, phone, address)
                   VALUES(
                   ?, ?, ?,?,?);
        """
        self._conn.executemany(sql, userid, password, name, phone, address)
        self._conn.commit()

    def insert_many(self0):
        data = [("lee", "1", "이순신", '010-1234-5678', '사당'),
                ("hong,"1","홍길동",'010-1234-4123','강남'),
                ("kang", "1", "강강처ㅏㄴ", '010-1234-7744', '분산')]
        # 데이타는 싸인다
        sql = """
            INSERT INTO Persons(
            personid, password, name, phone, address)
            VALUES(
            ?, ?, ?,?,?);
 """
        self._conn.executemany(sql, data)
        self._conn.commit()

    def fetch_one(self) -> object:
        sql = """
            SELECT * FROM Persons WHERE userid LIKE?;

        """
        cursor = self._conn.execute(sql, userid)
        row = cursor.fetchone()
        return row

    def fetch_many(self):
        pass

    def count_all(self) -> object:
        sql = """
            SELECT * FROM Persons;

        """
        cursor = self._conn.execute(sql)
        rows = cursor.fetchall()
        return rows

    def count_all(self) -> object:
        sql = """
                    SELECT COUNT(*) FROM Persons;

                """
        cursor = self._conn.execute(sql)
        row = cursor.fetchone()
        return row

    def update(self, password, userid):
        sql = """
            UPDATE Persons
            SET password = ?
            WHERE userid = ?;

        """
        self._conn.execute(sql, password, userid)

    def remove(self, userid):
        sql = '''
            DELETE FROM Persons WHERE Userid=?;

            '''

        self._conn.execute(sql, userid)



