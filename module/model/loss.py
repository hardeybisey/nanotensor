class Loss:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """
        y_true: arraylike of  actual values
        y_pred: arraylike of predicted values
        """
        loss = sum((e-p)**2 for e,p in zip(y_true,y_pred)) / len(y_true)
        return loss
    
    @staticmethod
    def binary_crossentropy(y_true, y_pred, eps=1e-15):
        """
        y_true: actual probability
        y_pred: predicted probability
        eps: small value to avoid log(0)
        """
        loss = -(y_true*y_pred.log() + (1-y_true)*(1-y_pred).log())
        return loss
