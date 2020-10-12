import torch
import numpy as np
import torch.nn.functional as F


class MC_Dropout:
    """
    MC dropout from the paper "Dropout as a Bayesian Approximation: 
    Representing Model Uncertainty in Deep Learning"
    
    https://arxiv.org/pdf/1506.02142.pdf
    """        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @staticmethod
    def mc_dropout_validate(model, criterion, testloader, T=100):
        """
        Validate on the whole test dataset
        """        
        model.train()
        loss = 0.
        accuracy =0.
        total = 0.
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, labels = data
                inputs = inputs.to(MC_Dropout.device)
                labels = labels.to(MC_Dropout.device)
                outputs = torch.zeros(inputs.size(0), 10).to(MC_Dropout.device)
                for _ in range(T):
                    outputs += model(inputs)
                outputs /= T
                loss += criterion(outputs, labels)
                score, predicted = torch.max(outputs, 1)
                accuracy += (labels == predicted).sum().item()
                total += labels.size()[0]

        print('{} Stochastic forward passes. Average MC Dropout Test loss: {:.3f}. MC Dropout Accuracy: {:.2f}%'.format(T,
            (loss / len(testloader)), (100 * accuracy / total)))       

    @staticmethod
    def predict_class(model, images, top_k):
        """
        Output the top k classes and their corresponding probability
        """        
        model.eval()
        with torch.no_grad():
            images = images.to(MC_Dropout.device)
            outputs = model(images)
            prob = F.softmax(outputs, dim=1)
            score, predicted= torch.topk(prob, top_k, dim=1)

        return score, predicted
        
    @staticmethod
    def mc_dropout_predict(model, images, T=10):
        """
        Output class logit score and class probability for all classes
        """
        class_logits = []
        class_prob = []
        model.train()
        with torch.no_grad():
            images = images.to(MC_Dropout.device)
            for _ in range(T):
                logits = model(images)
                prob = F.softmax(logits, dim=1)
                class_logits.append(logits.cpu().numpy())
                class_prob.append(prob.cpu().numpy())
        return np.array(class_logits), np.array(class_prob)
