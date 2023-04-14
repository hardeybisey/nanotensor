import numpy as np
class Loss:
    @classmethod
    def binary_crossentropy(cls, qq, pp, eps=1e-15):
        """
        p: predicted probability
        q: excpected probability
        """
        N = len(qq)
        constant =  1e-15#
        loss = -sum([q * p.log() + (1-q)*(1-p).log() for q,p in zip(qq,pp)])
        # loss = -sum(q[i] * p[i].log() for i in range(N))
        # loss = -sum((q[i] * p[i].log()) + ((1-q[i])*np.log(1-p[i])) for i in range(N))
        loss/N
        return loss
    
    def categorical_crossentropy(cls, q, p):
        """
        p: predicted probability
        q: excpected probability
        """
        N = len(q)
        loss = None
        loss = -sum([q[i]* p[i].log() for i in range(N)]) 
        return loss/N
    

    #     for i in range(N):
    #         for j in range(M):
    #             loss += -(q[j] * np.log(p[j])) + ((1-q[j])*np.log(1-p[j]))
    #     return loss
    

    # def cross_entropy(p, q):
    #     return -sum([p[i]*log(q[i]) for i in range(N)])