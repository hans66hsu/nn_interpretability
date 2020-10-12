import torch
import numpy as np
import torch.nn.functional as F


class Evidential_DL:
    """
    Evidential deep learning from the paper
    "Evidential Deep Learning to Quantify Classification Uncertainty"
    
    https://arxiv.org/pdf/1806.01768.pdf  
    Here we use the MSE loss which is mentioned as Eq. 5 in the paper
    """    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @staticmethod
    def kl_divergence(alpha_t):
        num_cls = alpha_t.size(1)
        beta = torch.ones([1,num_cls], device=Evidential_DL.device)
        S_alpha_t = torch.sum(alpha_t, dim=1, keepdim=True)
        KL = torch.sum((alpha_t - beta) * (torch.digamma(alpha_t) - torch.digamma(S_alpha_t)), dim=1, keepdim=True) +\
             torch.lgamma(S_alpha_t) - torch.sum(torch.lgamma(alpha_t), dim=1, keepdim=True) +\
             torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(torch.sum(beta, dim=1, keepdim=True))
        return KL
    
    @staticmethod
    def edl_mse_loss(outputs, labels, epoch_num, annealing_step):
        y_onehot = torch.zeros_like(outputs, device=Evidential_DL.device)
        y_onehot.scatter_(1, labels.unsqueeze(1), 1)

        evidence = F.relu(outputs)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        # logliklihood loss consists of prediction error and variance of the Dirichlet experiment
        lglh = torch.sum((y_onehot - alpha/S)**2, dim=1, keepdim=True) +\
               torch.sum(alpha * (S-alpha) / ((S**2)*(S+1.)), dim=1, keepdim=True)

        alpha_t = y_onehot + (1-y_onehot) * alpha
        KLreg = Evidential_DL.kl_divergence(alpha_t)
        annealing_coeff = min(1. , epoch_num / annealing_step)
        return torch.mean(lglh + annealing_coeff * KLreg)
    
    @staticmethod    
    def edl_validate(model, criterion, testloader):
        """
        Validate on the whole test dataset
        """
        model.train()
        loss = 0.
        accuracy =0.
        total = 0.
        uncertainty = 0.
        evidence = 0.
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, labels = data
                inputs = inputs.to(Evidential_DL.device)
                labels = labels.to(Evidential_DL.device)
                outputs = model(inputs)
                loss += criterion(outputs, labels, 1., annealing_step=10)
                evd = F.relu(outputs)
                alpha = evd + 1
                u = outputs.size(1) / torch.sum(alpha, dim=1, keepdim=True)
                evidence += evd.sum().item()
                uncertainty += u.sum().item()
                score, predicted = torch.max(outputs, 1)
                accuracy += (labels == predicted).sum().item()
                total += labels.size()[0]

        print('Average Evidential Deep Learning Test loss: {:.3f} | Accuracy: {:.2f}% | Evidence: ' 
              '{:.3f} | Uncertainty: {:.3f}'.format((loss / len(testloader)), (100 * accuracy / total), 
               evidence / total, uncertainty / total))    
        
    @staticmethod
    def predict_class(model, images, top_k):
        """
        Output the top k classes and their corresponding probability
        """
        model.eval()
        with torch.no_grad():
            images = images.to(Evidential_DL.device)
            outputs = model(images)
            evd = F.relu(outputs)
            alpha = evd + 1
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            score, predicted= torch.topk(prob, top_k, dim=1)

        return score, predicted
    
    @staticmethod 
    def edl_predict(model, images):
        """
        Output class logit score and class probability for all classes and uncertainty mass
        """
        class_logits = []
        class_prob = []
        class_uncertain = []
        model.eval()
        with torch.no_grad():
            images = images.to(Evidential_DL.device)
            logits = model(images)
            evd = F.relu(logits)
            alpha = evd + 1
            u = logits.size(1) / torch.sum(alpha, dim=1, keepdim=True)
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            class_logits.append(logits.cpu().numpy())
            class_prob.append(prob.cpu().numpy())
            class_uncertain.append(u.cpu().numpy())
        return np.array(class_logits), np.array(class_prob), np.array(class_uncertain)
