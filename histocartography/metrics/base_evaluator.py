
class BaseEvaluator:
    """
    Base interface class for evaluation metrics.
    """

    def __init__(self, cuda=False):
        """
        Base Evaluator constructor.

        Args:
            :param cuda: (bool) if cuda is available
        """
        self.cuda = cuda

    def __call__(self, logits, labels):
        """
        Evaluate
        Args:
            :param logits: (FloatTensor)
            :param labels: (LongTensor)
        """
        raise NotImplementedError("Implementation in sub classes.")
