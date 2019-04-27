from chap_2_saver.controller import TFcontroller
if __name__ == '__main__':
    c=TFcontroller()
    num1=5.0
    num2=10.0
    print('{}*{}={}'.format(num1,num2,c.calc(5.0,10.0)))