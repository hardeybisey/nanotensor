class Loss:
    @staticmethod
    def binary_crossentropy(pp, qq, eps=1e-15):
        """
        p: array-like of expected probability
        q: array-like of predicted probability
        eps: small value to avoid log(0)
        """
        n = len(pp)
        loss = -sum(p * (q+eps).log() + (1-p)*(1- (q+eps)).log() for q,p in zip(pp,qq))
        return loss/n