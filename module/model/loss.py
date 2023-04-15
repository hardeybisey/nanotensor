class Loss:
    @staticmethod
    def binary_crossentropy(y_true, y_pred, eps=1e-15):
        """
        pp: array-like of predicted probability
        qq: array-like of expected probability
        eps: small value to avoid log(0)
        """
        n = len(y_true)
        # for p,q in zip(pp,qq):
        #     positive = q * (p+eps).log()
        #     negative = (1-q) * ((1- (p+eps)).log())
        #     loss += - positive + negative
        loss = sum(q * (p+eps).log() + (1-q) *((1- (p+eps)).log()) for p,q in zip(y_true,y_pred))
        return -loss/n