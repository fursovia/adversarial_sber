import torch
from allennlp.training.metrics import Average


class FixedPerplexity(Average):
    def get_metric(self, reset: bool = False):
        """
        # Returns

        The accumulated perplexity.
        """
        average_loss = super().get_metric(reset)
        if average_loss == 0:
            perplexity = 0.0
            return perplexity

        if isinstance(average_loss, float):
            average_loss = torch.tensor(average_loss)

        # Exponentiate the loss to compute perplexity
        perplexity = float(torch.exp(average_loss))

        return perplexity
