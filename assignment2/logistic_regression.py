import torch
import torch.nn as nn
from tqdm import tqdm


def nonprivate_logistic_regression(dset_loader, num_epochs, learning_rate,
        lmbda, seed=None):
    if seed:
        torch.manual_seed(seed)
    num_pixels = dset_loader.dataset.num_pixels
    model = nn.Linear(num_pixels, 1, bias=False)
    # note that the model outputs the log probabiilty of the label being positive
  
    # Loss and optimizer
    # nn.CrossEntropyLoss() computes softmax internally
    criterion = nn.BCEWithLogitsLoss()  # binary cross entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, 
        momentum=0.9, weight_decay=lmbda)  

    # Train the model
    num_train_examples = len(dset_loader.dataset)
    total_step = len(dset_loader)
    for epoch in tqdm(range(num_epochs)):
        train_loss = 0.
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(dset_loader):
            # Reshape images to (batch_size, input_size)
            images = images.reshape(-1, 28*28)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            train_loss += loss * (len(images) / float(num_train_examples))
            predicted = (outputs.squeeze() > 0.).long()
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model.state_dict()




if __name__ == '__main__':
    from utils import get_data_loaders
    # hyperparams
    batch_size = 64
    num_epochs = 100
    learning_rate = 1e-1
    lmbda = 5e-3
    data_seed = 0
    num_train = 1000
  
    # load data
    loaders, _ = get_data_loaders(data_seed, batch_size, num_train)
  
    # train model
    nonprivate_params = \
            nonprivate_logistic_regression(loaders['train'], num_epochs,
                    learning_rate, lmbda)
