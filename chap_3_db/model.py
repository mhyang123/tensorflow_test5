from chap_3_db.db import Database


class Model:
    def __init__(self):
        pass

    def create_db(self) -> str:
        db = Database()
        db.create()
        db.insert_many()
        count = db.count_all()
        print('DB에 등록된 회원수: {}', count)
        return count