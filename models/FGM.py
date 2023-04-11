import torch
class FGM():
    """
    Define the adversarial training method FGM, which perturbs the model embedding parameters
    """
    def __init__(self, model, epsilon=0.25,):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self, emb_name='word_embeddings'):
        """
        Obtaining an adversarial sample
        :param emb_name: Parameter names for embedding in the model
        :return:
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        """
        Recovering the original parameters of the model
        :param emb_name:  Parameter names for embedding in the model
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}