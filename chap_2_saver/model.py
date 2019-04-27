class TFModel:
    def __int__(self):
        self.num1= 0.0
        self.num2= 0.0

    @property
    def num1(self): return self._num1
    @num1.setter
    def num1(self): self.num1 = num1

    @property
    def num2(self):return self._num2
    @num1.setter
    def num2(self):self.num1=num2

    def exec(self)->float :
        num1=self._num1
        num2 = self._num2
        var = tf.Variable([num1], dtype=tf.float32)
        con = tf.constant([num2], dtype=tf.float32)
        session=tf.Session()
        init = tf.global_variables_initializer()
        # titanic와 달리 답안이 없다.그러므로 비지도 학습니다.
        session.run(init)
        result=session.run(var * con)#모델
        saver=tf.train.Saver()
        saver.save(session,'./data/model.ckpt')#session는 항상 바꾸어지는 값이다
        return result


