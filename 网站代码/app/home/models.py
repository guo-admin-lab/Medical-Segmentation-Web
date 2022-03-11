from sqlalchemy import Column, Integer, String, Text
from app import db


class Feedback(db.Model):
    __tablename__ = 'Feedback'

    id = Column(Integer, primary_key=True)
    username = Column(String)
    casename = Column(String)
    modelname = Column(String)
    description = Column(Text)
    metrics = Column(String)
    time = Column(String)

    # __mapper_args__ = {
    #     "order_by": id.desc()
    # }

    def __init__(self, username, casename, modelname, description, metrics, time):
        self.username = username
        self.casename = casename
        self.modelname = modelname
        self.description = description
        self.metrics = metrics
        self.time = time

    def __repr__(self):
        print('hhh')


class Record(db.Model):
    __tablename__ = 'Record'

    id = Column(Integer, primary_key=True)
    username = Column(String)
    image = Column(String)
    prediction = Column(String)
    time = Column(String)
    name = Column(String)  # 病例名称
    modelname = Column(String)  # 使用模型名称
    description = Column(Text)  # 病例分析

    # __mapper_args__ = {
    #     "order_by": id.desc()
    # }

    def __init__(self, username, image, prediction, time, name, modelname, description):
        self.username = username
        self.image = image
        self.prediction = prediction
        self.time = time
        self.name = name
        self.modelname = modelname
        self.description = description

    def __repr__(self):
        return f'name: {self.name}\n description: {self.description}'


class Database(db.Model):
    __tablename__ = 'Database'

    id = Column(Integer, primary_key=True)
    username = Column(String)
    name = Column(String)
    type = Column(String)
    time = Column(String)

    # __mapper_args__ = {
    #     "order_by": id.desc()
    # }

    def __init__(self, username, name, type, time):
        self.username = username
        self.name = name
        self.type = type
        self.time = time

    def __repr__(self):
        return '打印啥呢你'


class PersonalInfo(db.Model):
    __tablename__ = 'PersonalInfo'

    id = Column(Integer, primary_key=True)
    username = Column(String)
    email = Column(String)
    name = Column(String)
    position = Column(String)
    work_unit = Column(String)
    city = Column(String)
    province = Column(String)
    introducation = Column(Text)

    def __init__(self, username, email, name=None, position=None, work_unit=None, city=None, province=None, introducation=None):
        self.username = username
        self.email = email
        self.name = name
        self.position = position
        self.work_unit = work_unit
        self.city = city
        self.province = province
        self.introducation = introducation

    def __repr__(self):
        return 'hhh'