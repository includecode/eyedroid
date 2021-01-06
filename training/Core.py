import torch
import NetworkConf
import Extract
import  BuildAndLoss
import TrainSingleBatch
import TrainAllBatches
import  TrainMultipleEpochs



# Preparing data using PyTorch
train_set = Extract.ExtractData()
train_set = train_set.extract_data()

# Initialising the model architecture by defining Network class and instructions for forward pass
network = NetworkConf.Network()

# PyTorch DataLoader class for loading our data and initiating for the Forward Class
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
batch = next(iter(train_loader)) # Getting a batch
images, labels = batch

# Build the model & Calculating the loss
loss = BuildAndLoss.BuildAndLossClass()
loss = loss.build_and_compute_loss(network, images, labels)

# ------------Training on Single batch--------------
# Calculating the loss
# Calculating the Gradients
# Updating the Weights
# Retraining
train_single = TrainSingleBatch.Train_single_batch()
train_single.train(train_set)


# Training with all batches (= One Epoch) 
# Running it in a loop instead of doing single batch-wise.

# We have 60,000 samples in our training set, we will have 60,000 / 100 = 600 iterations done in one go.
train_all = TrainAllBatches.Train_all_batches()
train_all.train(train_set)


####TRAINING ALL EPOCHS FOR MORE ACCURENCY
train_epochs = TrainMultipleEpochs.Train_multiple_epochs()
train_epochs = train_epochs.train(train_set)

#############################################################################
############################### T   E   S   T   #############################
#############################################################################
# get Predictions for ALL Samples

@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch
 
        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds

with torch.no_grad():
    prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
    train_preds = get_all_preds(network, prediction_loader)

def get_num_correct(preds, labels):
            return preds.argmax(dim=1).eq(labels).sum().item()
preds_correct = get_num_correct(train_preds, train_set.targets)
 
print('total correct:', preds_correct)
print('accuracy:', preds_correct / len(train_set))