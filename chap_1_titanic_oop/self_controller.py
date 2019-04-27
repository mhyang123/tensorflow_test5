
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from chap_1_titanic_oop.model import TitanicModel
from chap_1_titanic_oop.view import TitanicView


class TitanicController:
    def __init__(self):
        self._ m =TitanicModel()
        self._ v =TitanicView()
        self._contex t ='./data/'
        self._trai n =self.create_train()

    def create_train(self )- >object: m=self._m
        m.context = self._context
        m.fname='train. c sv'
        t1=m.new_d f rame()
        m.fname='test.c s v'
        t2=m.new_d f rame()
        train=m.hook_ p rocess(t1,t2)
        print(' ---1----')
        print(train.columns)
        print(' ---2----')
        print(train.head())
        return  train

    def ctreate_model(self) ->object:
        train =self._tr ain###????이  유
        model=train.d r op('Survived',axis = 1 )
        print(' ---model info----')
        print(model.info)
        return model

    def ctreate_dummy(self) ->object:
        train =self._tr ain ###????  유
        dummy=train.d r op('Survived')

        return dummy
    @st

    ethod
    def create_random_variable(train,X_featur es,Y_featur es)->[]:
        the_X_features=X_featu r es
        the_Y_features=Y_featu r es
        train2,test2=tr ain_t e st_split(train,test_siz e=0.3,random_s tate=0)
        train_X=train2[ t he_X_features]
        train_Y = train2[the_Y_features]
        test_X=test2[t h e_X_features]
        test_Y = test2[the_Y_features]
        return [train_X,test_Y,t rain_Y, test_Y]

    # ****
    # 러닝
    # ****

    def test_random_variable(self)->str:
        train=self._t r ain()
        x_feature=['Pclas s ','Sex','E mbarke d']
        y_feature = ['Survived']
        random_variables=self.cr e ate_random_variables(train,X_featur es,Y_featur es)
        accuracy=self.ac c uracy_by_decision_tree(
            random_variables[0],
            random_variables[1],
            random_variables[2],
            random_variables[3]
        )
        return accuracy
'''
   


@staticmethod
    def accuracy_by_decision_tree(train_X,test_Y,train_Y,test_Y)->str:
         tree_model=DecisionTreeClassifier()
         tree_model.fit(train_X.values,train_Y.values)
         dt_prdiction=tree_model.predict(test_X)
         accuracy =metrics.accuracy_score(dt_prdiction,test_Y)
         return  accuracy

'''


def test_by_sklearn(self):
    print('------------사이킷런을 활용한 검증---------------')
    model = self.create_model()
    dummy = self.create_dummy()
    m = self._m
    print('---------------- KNN 방식 정확도 ----------------')
    accuracy = m.accuracy_by_knn(model, dummy)
    print(' {} %'.format(accuracy))

    print('---------------- 결정트리 방식 정확도 ----------------')
    accuracy = m.accuracy_by_dtree(model, dummy)
    print(' {} %'.format(accuracy))

    print('---------------- 랜덤포레스트 방식 정확도 ----------------')
    accuracy = m.accuracy_by_rforest(model, dummy)
    print(' {} %'.format(accuracy))

    print('---------------- 나이브베이즈 방식 정확도 ----------------')
    accuracy = m.accuracy_by_nb(model, dummy)
    print(' {} %'.format(accuracy))

    print('---------------- SVM 방식 정확도 ----------------')
    accuracy = m.accuracy_by_svm(model, dummy)
    print(' {} %'.format(accuracy))


def submit(self):
    m = self._m
    model = self.create_model()
    dummy = self.create_dummy()
    test = m.test
    test_id = m.test_id
    clf = SVC()
    clf.fit(model, dummy)
    prediction = clf.predict(test)
    submission = pd.DataFrame(
        {'PassengerId': test_id,
         'Survived': prediction
         }
    )
    print(submission.head())
    submission.to_csv(m.context + 'submission.csv', index=False)




























