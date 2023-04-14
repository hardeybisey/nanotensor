class Loss:
    @classmethod
    def bianaryentropy(cls, ytrue, ypred):
        loss = (-ytrue * np.log(ypred)) + ((1-ytrue)*np.log(1-ypred))
        return loss