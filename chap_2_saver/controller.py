from chap_2_saver.model import TFModel
#from chap_2_saver.view import TFView

class TFcontroller:
    def __int__(self):
        self.m=TFModel()
        self.v=TFView()
    def calc(self,num1,num2)->float:
        m=self._m
        m.num1=num1
        m.num2=num2
        return m.exec()


