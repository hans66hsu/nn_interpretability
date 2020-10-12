import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class ModelTrainer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def train(model, criterion, optimizer, trainloader, epochs=5):
        criterion = criterion.to(ModelTrainer.device)
        model = model.to(ModelTrainer.device)
        model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs = inputs.to(ModelTrainer.device)
                labels = labels.to(ModelTrainer.device)
                inputs.requires_grad = True

                optimizer.zero_grad()

                outputs = model(inputs).to(ModelTrainer.device)
                loss = criterion(outputs, labels).to(ModelTrainer.device)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99 or i == len(trainloader) - 1:
                    _, predicted = torch.max(outputs.data, 1)
                    total = labels.size(0)
                    accuracy = (predicted == labels).sum().item() / total
                    loss = running_loss / 100

                    print('Epoch: %d | Batch: %d | Loss: %.3f | Accuracy: %.3f' %
                          (epoch + 1, i + 1, loss, accuracy))
                    running_loss = 0.0

        print('Finished Training')

        return model

    @staticmethod
    def train_mnist_gan(generator, discriminator, dataloader, lr, latent_dim, epochs=100):
        generator = generator.to(ModelTrainer.device)
        discriminator = discriminator.to(ModelTrainer.device)

        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
        adversarial_loss = torch.nn.BCELoss().to(ModelTrainer.device)

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        batches_done = 0

        gen_losses = []
        disc_losses = []

        for epoch in range(epochs):
            for i, (imgs, _) in enumerate(dataloader):
                valid = Variable(Tensor(imgs.size(0), 1).fill_(0.9), requires_grad=False).to(ModelTrainer.device)
                fake = Variable(Tensor(imgs.size(0), 1).fill_(0.1), requires_grad=False).to(ModelTrainer.device)
                real_imgs = Variable(imgs.type(Tensor)).to(ModelTrainer.device)

                gen_optimizer.zero_grad()
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))).to(ModelTrainer.device)

                gen_imgs = generator(z).to(ModelTrainer.device)
                predicted_true = discriminator(gen_imgs).to(ModelTrainer.device)
                g_loss = adversarial_loss(predicted_true, valid).to(ModelTrainer.device)
                g_loss.backward()
                gen_optimizer.step()

                disc_optimizer.zero_grad()

                predicted_true = discriminator(real_imgs).to(ModelTrainer.device)
                real_loss = adversarial_loss(predicted_true, valid).to(ModelTrainer.device)

                predicted_false = discriminator(gen_imgs.detach()).to(ModelTrainer.device)
                fake_loss = adversarial_loss(predicted_false, fake).to(ModelTrainer.device)

                disc_loss = (real_loss + fake_loss) / 2
                disc_loss.to(ModelTrainer.device)

                disc_loss.backward()
                disc_optimizer.step()

                if i % 100 == 0:
                    print(f'Epoch: {epoch+1}/{epochs} | Batch: {batches_done % len(dataloader)}/{len(dataloader)} | Discriminator Loss: {disc_loss.item():.3f} | Generator Loss: {g_loss.item():.3f}')
                    disc_losses.append(disc_loss.item())
                    gen_losses.append(g_loss)

                batches_done += 1

        return gen_losses, disc_losses

    @staticmethod
    def train_evidence(model, criterion, optimizer, trainloader, epochs=30):
        criterion = criterion.to(ModelTrainer.device)
        model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            running_uncertainty = 0.0
            running_evidence = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs = inputs.to(ModelTrainer.device)
                labels = labels.to(ModelTrainer.device)


                inputs.requires_grad = True

                optimizer.zero_grad()

                outputs = model(inputs).to(ModelTrainer.device)
                loss = criterion(outputs, labels, epoch, annealing_step=10).to(ModelTrainer.device)

                evidence = F.relu(outputs)
                alpha = evidence + 1
                u = outputs.size(1) / torch.sum(alpha, dim=1, keepdim=True)


                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_uncertainty += u.mean().item()
                running_evidence += torch.sum(evidence, dim=1).mean().item()
                if i % 100 == 99 or i == len(trainloader) - 1:
                    _, predicted = torch.max(outputs.detach(), 1)
                    total = labels.size(0)
                    accuracy = (predicted == labels).sum().item() / total
                    match = (predicted == labels).float()

                    loss = running_loss / 100
                    uncertainty = running_uncertainty / 100
                    evidence = running_evidence / 100

                    print('Epoch: {:d} | Batch: {:d} | Loss: {:.3f} | Accuracy: {:.1f}% | '
                          'Evidence: {:.1f} | Uncertainty: {:.3f}'.format
                          (epoch + 1, i + 1, loss, 100 * accuracy, evidence, uncertainty))
                    running_loss = 0.0
                    running_uncertainty = 0.0
                    running_evidence = 0.0
        print('Finished Training')

        return model           
                        
    @staticmethod
    def validate(model, criterion, testloader):
        running_loss = 0.
        accuracy = 0.
        total = 0.
        model.eval()
        criterion = criterion.to(ModelTrainer.device)
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, labels = data
                inputs = inputs.to(ModelTrainer.device)
                labels = labels.to(ModelTrainer.device)
                outputs = model(inputs).to(ModelTrainer.device)
                score, predicted = torch.max(outputs, 1)
                accuracy += (labels == predicted).sum().item()
                total += labels.size()[0]
                loss = criterion(outputs, labels).to(ModelTrainer.device)

                running_loss += loss.item()

        print('Average Test loss: {:.3f}. Accuracy: {:.2f}%'.format(
            (running_loss / len(testloader)), (100 * accuracy / total)))
