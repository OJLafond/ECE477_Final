# Function to load the training data that will be used to learn the architecture and the test data used for evaluation
def load_data_train(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    y_train = np.where(y_train == 'Relax', 0, np.where(y_train == 'Stress', 1, y_train)).astype(int)
    y_test = np.where(y_test == 'Relax', 0, np.where(y_test == 'Stress', 1, y_test)).astype(int)

    return X_train, y_train, X_test, y_test

import numpy as np
import torch

def to_numpy_safe(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def loadData(self, X_train, y_train, X_test, y_test, mode='train', fold=None):
    if mode == 'train':
        # Ensure inputs are NumPy arrays
        X_train = to_numpy_safe(X_train)
        y_train = to_numpy_safe(y_train)

        self.X_train, self.y_train, self.X_validation, self.y_validation = load_data_train(X_train, y_train)

        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train.reshape(-1), dtype=torch.long)

        self.X_validation = torch.tensor(self.X_validation, dtype=torch.float32)
        self.y_validation = torch.tensor(self.y_validation.reshape(-1), dtype=torch.long)

        self.traindata = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        self.trainloader = torch.utils.data.DataLoader(self.traindata, batch_size=self.batch_size, shuffle=True)

        self.validationdata = torch.utils.data.TensorDataset(self.X_validation, self.y_validation)
        self.validationloader = torch.utils.data.DataLoader(self.validationdata, batch_size=self.batch_size, shuffle=False)

    elif mode == 'test':
        X_test = to_numpy_safe(X_test)
        y_test = to_numpy_safe(y_test)

        self.X_test = X_test.astype(float)
        self.y_test = np.where(y_test == 'Relax', 0, np.where(y_test == 'Stress', 1, y_test)).astype(int)

        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test.reshape(-1), dtype=torch.long)

        self.testdata = torch.utils.data.TensorDataset(self.X_test, self.y_test)
        self.testloader = torch.utils.data.DataLoader(self.testdata, batch_size=self.batch_size, shuffle=False)

SDNN.loadData = loadData

# initializing the structure
# four main components and their corresponding masks
# We define weight matrices, and masks, for connection between input neurons to hidden neurons,
# hidden neurons to hidden neurons, hidden neurons to output neurons, and input neurons to output neurons
# We also define bias terms for hidden neurons and output neurons

def structureInit(self, load=False, sparse=True, ratio=0.2, file=None):
    # input to hidden
    self.w1 = torch.randn(self.in_num, self.max_size) * 0.1
    self.m1 = torch.zeros(self.in_num, self.max_size)
    # hidden to hidden
    self.w2 = torch.randn(self.max_size, self.max_size) * 0.1
    self.m2 = torch.ones(self.max_size, self.max_size)
    # hidden to output
    self.w3 = torch.randn(self.max_size, self.out_num) * 0.1
    self.m3 = torch.zeros(self.max_size, self.out_num)
    # input to output
    self.w4 = torch.randn(self.in_num, self.out_num) * 0.1
    self.m4 = torch.ones(self.in_num, self.out_num)

    self.b1 = torch.zeros(1, self.max_size)
    self.b2 = torch.zeros(1, self.out_num)

    self.w1.requires_grad = True
    self.w2.requires_grad = True
    self.w3.requires_grad = True
    self.w4.requires_grad = True
    self.b1.requires_grad = True
    self.b2.requires_grad = True

    self.params = {'w1': self.w1, 'w2': self.w2, 'w3': self.w3, 'w4': self.w4,
                   'm1': self.m1, 'm2': self.m2, 'm3': self.m3, 'm4': self.m4,
                  }

    self.criterion = nn.CrossEntropyLoss()
    # your code is here
    # clarification: predefined self.params are intended for different purpose
    # don't pass it into the optimizer
    self.optimizer = optim.SGD([self.w1, self.w2, self.w3, self.w4, self.b1, self.b2], lr=0.001, momentum=0.9, weight_decay=1e-4, nesterov=True)

    # starting from scratch
    # at first, we have init_size active hidden neurons
    # neuronMask is a helper function defined in utils.py
    # This function sets the appropriate number of mask values equal to 1
    # note that in the initialization step, we only activate connections between input to hidden (m1)
    # and hidden to output (m3), and biases for hidden neurons (b1)
    if load == False:
        self.active_index = list(range(self.init_size))
        if sparse:
            self.m1.data = neuronMask(self.m1.data, self.active_index, sparse=True, ratio=ratio)
        else:
            self.m1.data = neuronMask(self.m1.data, self.active_index)
        self.m3.data = neuronMask(self.m3.data, self.active_index, 'out')
        self.b1.data = neuronMask(self.b1.data, self.active_index)

    # loading from a pretrained model
    # we have to load all the parameters of the model, including the index of active neurons
    # and all the learned weight, bias, and mask matrices
    else:
        checkpoint = torch.load(file, weights_only=True)
        self.active_index = checkpoint['active_index']
        self.w1.data = checkpoint['state_dict']['w1']
        self.m1.data = checkpoint['state_dict']['m1']
        self.w2.data = checkpoint['state_dict']['w2']
        self.m2.data = checkpoint['state_dict']['m2']
        self.w3.data = checkpoint['state_dict']['w3']
        self.m3.data = checkpoint['state_dict']['m3']
        self.w4.data = checkpoint['state_dict']['w4']
        self.m4.data = checkpoint['state_dict']['m4']
        self.b1.data = checkpoint['state_dict']['b1']
        self.b2.data = checkpoint['state_dict']['b2']
        self.epoch = checkpoint['epoch']
        self.best_acc = checkpoint['best_acc']
        self.now_acc = checkpoint['now_acc']
        self.optimizer.load_state_dict(checkpoint['optimizer'])


SDNN.structureInit = structureInit

def train(self, duration=10, folder_to_save='tmp'):
    for epoch in range(self.epoch, self.epoch+duration): #loop over the dataset multiple times based on #epochs
        running_loss = 0.0
        # reading the data using the data loaders defined earlier
        for i, data in enumerate(self.trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs.requires_grad_(True)
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            # computing the running loss
            running_loss += loss.item()

            # updating the weight matrices
            self.w1.data = self.w1.data * self.m1.data
            self.w2.data = self.w2.data * self.m2.data
            self.w3.data = self.w3.data * self.m3.data
            self.w4.data = self.w4.data * self.m4.data

        # computing the train accuracy
        total = 0
        correct = 0
        for i, data in enumerate(self.trainloader, 0):
            inputs, labels = data
            #inputs = inputs.view(inputs.size(0), -1)
            outputs = self.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.data).sum()
            total += labels.size(0)
        train_acc = correct * 1. / total

        # computing the validation accuracy
        total = 0
        correct = 0
        for i, data in enumerate(self.validationloader, 0):
            inputs, labels = data
            outputs = self.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.data).sum()
            total += labels.size(0)
        validation_acc = correct * 1. /total
        self.now_acc = validation_acc

        # saving the model if the validation accuracy is better than the current best validation accuracy
        # of the pruned model
        # we use the save_checkpoint function defined later to save the model and all the important parameters
        # you can change the filename as you wish
        if (validation_acc > self.best_acc_prune) and (self.flag == 1):
            self.best_acc_prune = validation_acc
            self.save_checkpoint({
                'epoch': epoch,
                'best_acc': self.best_acc,
                'now_acc': self.now_acc,
                'state_dict': {
                    'w1': self.w1.data,
                    'm1': self.m1.data,
                    'w2': self.w2.data,
                    'm2': self.m2.data,
                    'w3': self.w3.data,
                    'm3': self.m3.data,
                    'w4': self.w4.data,
                    'm4': self.m4.data,
                    'b1': self.b1.data,
                    'b2': self.b2.data,
                },
                'active_index': self.active_index,
                'optimizer': self.optimizer.state_dict(),
            }, False, folder_to_save, filename='_prune.pth.tar')

        if (validation_acc > self.best_acc):
            self.best_acc = validation_acc
            self.save_checkpoint({
                'epoch': epoch + 1,
                'best_acc': self.best_acc,
                'now_acc': self.now_acc,
                'state_dict': {
                    'w1': self.w1.data,
                    'm1': self.m1.data,
                    'w2': self.w2.data,
                    'm2': self.m2.data,
                    'w3': self.w3.data,
                    'm3': self.m3.data,
                    'w4': self.w4.data,
                    'm4': self.m4.data,
                    'b1': self.b1.data,
                    'b2': self.b2.data,
                },
                'active_index': self.active_index,
                'optimizer': self.optimizer.state_dict(),
            }, True, folder_to_save)
        else:
            self.save_checkpoint({
                'epoch': epoch,
                'best_acc': self.best_acc,
                'now_acc': self.now_acc,
                'state_dict': {
                    'w1': self.w1.data,
                    'm1': self.m1.data,
                    'w2': self.w2.data,
                    'm2': self.m2.data,
                    'w3': self.w3.data,
                    'm3': self.m3.data,
                    'w4': self.w4.data,
                    'm4': self.m4.data,
                    'b1': self.b1.data,
                    'b2': self.b2.data,
                },
                'active_index': self.active_index,
                'optimizer': self.optimizer.state_dict(),
            }, False, folder_to_save)
        print('Epoch: %d, Training accuracy: %f, Validation accuracy: %f'
              % (epoch, train_acc, validation_acc))

        m1,m2,m3,m4,m_all = self.displayConnection(display=False)

    self.epoch += duration


SDNN.train = train

# cell division function, we have the options between activation based, gradient-based, and random cell-division
# We normally use activation-based (duplicating the cell with the highest activation value) or
# random cell division (randomly selecting a hidden cell to be duplicated)
# We can make this decisions either by looking at the full data, or a batch of data
# Other than mode, the other inputs are num (shows number of neurons to be duplicated)
# full_data that shows whether or not to use the full data for neuron selection
# and if full data is flase, size shows how many batches to use to compute the neuron actications

def cellDivision(self, mode='acti', num=1, full_data=False, size=1):
    '''
    Function: add neurons.
    Arguments:
        mode: 'acti' activation-based,'grad' gradient-based, 'rand' random
        num: number of neurons added each time
        full_data: whether to use full data to decide which neuron to split
        size: if full_data=False, number of batches used to decide which neuron to split
    '''

    # computing the hidden activation values, either by using the whole data or several batches of data
    # we sum up the hidden activations for several batches
    if mode == 'acti':
        activation = np.zeros(self.max_size)
        if full_data:
            for i, data in enumerate(self.trainloader, 0):
                inputs,_ = data
                self.forward(inputs)
                activation += torch.sum(torch.abs(self.hidden.data), 0)
        else:
            loader = iter(self.trainloader)
            for i in range(size):
                inputs,_ = next(loader)
                self.forward(inputs)
                activation += torch.sum(torch.abs(self.hidden.data), 0).cpu().numpy()

        # selecting the neurons with the highest activations to be duplicated
        # we select 'num' neurons to be duplicated
        max_index_arr = np.flip(np.argsort(activation)[-num:], axis=0)
    elif mode == 'grad':

        # selecting the neurons to be activated based on the hidden gradients
        # we did not use this method in the final experiments of the paper
        # however, it is worth exploring
        # we use the function badwardGrad defined later to compute gradients
        activation = np.zeros(self.max_size)
        if full_data:
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.backwardGrad(outputs)
                activation += torch.sum(self.hidden.grad.data, 0)
        else:
            loader = iter(self.trainloader)
            for i in range(size):
                inputs, labels = loader.next()
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                self.backwardGrad(outputs)
                activation += torch.sum(self.hidden.grad.data, 0)
        max_index_arr = np.flip(np.argsort(activation)[-num:], axis=0)
    elif mode == 'rand':
        # selection 'num' neurons random from active neurons
        max_index_arr = np.random.choice(self.active_index, size=num, replace=False)

    # after selecting the neuron to be duplicated, we duplicate that neuron and its connections,
    # and add noise to weights of the new added neuron

    for max_index in max_index_arr:
        # we add the index at the end of active_index list of active neurons
        add_index = len(self.active_index)
        # current size
        self.cur_size = add_index
        print('Max index: %d' %max_index)

        # we only add a new neurons if the number of neurons will be less than the maximum number of neurons
        # set at the beginning
        if add_index < self.max_size:
            print('Adding neuron: %d' %add_index)
            python_max_index = int(max_index)  # Convert to Python int
            if python_max_index in self.active_index:
                self.active_index.insert(self.active_index.index(python_max_index), add_index)
                # duplicating the masks
                self.m1.data[:, add_index] = self.m1.data[:, python_max_index]
                self.m2.data[:, add_index] = self.m2.data[:, python_max_index]
                self.m3.data[add_index, :] = self.m3.data[python_max_index, :]

                # duplicating the weight matrices and adding noise
                self.w1.data[:, python_max_index] = self.w1.data[:, python_max_index]
                self.w1.data[:, add_index] = self.w1.data[:, python_max_index] + torch.randn(self.in_num) * 0.01
                self.w2.data[:, add_index] = self.w2.data[:, python_max_index] + torch.randn(self.max_size) * 0.01
                self.w3.data[add_index, :] = self.w3.data[python_max_index, :] + torch.randn(self.out_num) * 0.01
                self.b1.data[:, add_index] = self.b1.data[:, python_max_index]
            else:
                print(f"Value {python_max_index} not found in active_index. Skipping insertion due to error.")

    # updating the weight matrices
    self.w1.data = self.w1.data * self.m1.data
    self.w2.data = self.w2.data * self.m2.data
    self.w3.data = self.w3.data * self.m3.data
    self.w4.data = self.w4.data * self.m4.data
    self.displayConnection()


SDNN.cellDivision = cellDivision

def addConnection(self, mode='grad', percentile={'m2':90, }, size=1, full_data=False):
    '''
    Function: add connections.
    Arguments:
        mode: 'corr' correlation-based, 'grad' gradient-based, 'rand' random
        percentile: top-k percentile of connections are added
    '''
    print('\nAdding connection...')
    self.flag = 0

    cov_mat = {
        'm1': np.zeros([self.in_num, self.max_size]),
        'm2': np.zeros([self.max_size, self.max_size]),
        'm3': np.zeros([self.max_size, self.out_num]),
        'm4': np.zeros([self.in_num, self.out_num]),
    }

    # gradient-based growth
    # we use the backwardGrad function to compute gradients
    if mode == 'grad':
        loader = iter(self.trainloader)
        for i in range(size):
            inputs, labels = next(loader)
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.backwardGrad(outputs)

            cov_mat_m1 = torch.mm(inputs.T, self.hidden.grad)
            cov_mat_m2 = torch.mm(self.hidden.T, self.hidden.grad)
            cov_mat_m3 = torch.mm(self.hidden.T, outputs.grad)
            cov_mat_m4 = torch.mm(inputs.T, outputs.grad)

            # add to covariance matrix values cov_mat_m1, cov_mat_m2, cov_mat_m3, cov_mat_m4
            cov_mat['m1'] = np.add(cov_mat['m1'], cov_mat_m1.detach().numpy())
            cov_mat['m2'] = np.add(cov_mat['m2'], cov_mat_m2.detach().numpy())
            cov_mat['m3'] = np.add(cov_mat['m3'], cov_mat_m3.detach().numpy())
            cov_mat['m4'] = np.add(cov_mat['m4'], cov_mat_m4.detach().numpy())

    elif mode == 'rand':
            cov_mat['m1'][:, :self.cur_size] = np.random.rand(self.in_num, self.cur_size)
            cov_mat['m2'][:self.cur_size, :self.cur_size] = np.random.rand(self.cur_size, self.cur_size)
            cov_mat['m3'][:self.cur_size, :] = np.random.rand(self.cur_size, self.out_num)
            cov_mat['m4'] = np.random.rand(self.in_num, self.out_num)

    for i in percentile:
        if self.scheme == "C" and i == 'm2':
          mask = np.zeros_like(cov_mat['m2'])
          for j in range(self.max_size - 1):
            mask[j, j + 1] = 1
            cov_mat[i] *= mask

        if len(np.nonzero(cov_mat[i])[0]) == 0:
            threshold = 0
        else:
            threshold = np.percentile(cov_mat[i][np.nonzero(cov_mat[i])], percentile[i])
        self.params[i].data[torch.Tensor(cov_mat[i])>threshold] = 1

    self.forwardMask()
    self.displayConnection()

    # self.m1, self.m2, self.m3, self.m4 are masks for correspondings weights.
    # they are float tensors containing 1. and 0. values
    # update weights masking out corresponding values.
    # Impprtant: For weights and masks tensors in calculation, call their .data() property
    # to prevent tracking gradients on these tensors by torch autograd system

    self.w1.data = self.w1.data * self.m1.data
    self.w2.data = self.w2.data * self.m2.data
    self.w3.data = self.w3.data * self.m3.data
    self.w4.data = self.w4.data * self.m4.data



SDNN.addConnection = addConnection

def pruneConnections(self, prune_ratio=0.2):
    """
    Prune connections with the smallest magnitude weights (among active connections).
    Only considers weights where the corresponding mask == 1.
    """
    print("\nPruning connections...")
    self.flag = 1  # Indicates a pruning step occurred

    for key in ['m1', 'm2', 'm3', 'm4']:
        weight = self.params[key.replace('m', 'w')]
        mask = self.params[key]

        # Only consider currently active connections (mask == 1)
        active_weights = torch.abs(weight.data[mask.data == 1])
        if active_weights.numel() == 0:
            continue  # No active weights to prune

        # Determine pruning threshold
        threshold = torch.quantile(active_weights, prune_ratio)

        # Prune (set mask to 0 where weight < threshold)
        to_prune = (torch.abs(weight.data) < threshold) & (mask.data == 1)
        mask.data[to_prune] = 0
        weight.data *= mask.data  # Apply the new mask

    self.displayConnection()

SDNN.pruneConnections = pruneConnections

def displayAcc(self):
    """computing and displaying the train and test accuracy"""
    total = 0
    correct = 0
    for i, data in enumerate(self.trainloader, 0):
        inputs, labels = data
        outputs = self.forward(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.data).sum()
        total += labels.size(0)
    print('Train: %d/%d' %(correct, total))
    train_acc = correct * 1. / total

    total = 0
    correct = 0

    for i, data in enumerate(self.validationloader, 0):
      inputs, labels = data
      outputs = self.forward(inputs)
      _, predicted = torch.max(outputs.data, 1)
      correct += (predicted == labels.data).sum()
      total += labels.size(0)
    print('Validation: %d/%d' %(correct, total))
    validation_acc = correct * 1. / total

    print('Train accuracy: %f, Test accuracy: %f'
          % (train_acc, validation_acc))
    return validation_acc


SDNN.displayAcc = displayAcc

# create folder to store checkpoints
os.makedirs('record_full', exist_ok=True)

def compute_sparsity(sdnet):
    total = 0
    zeros = 0
    for param in [sdnet.w1, sdnet.w2, sdnet.w3, sdnet.w4, sdnet.b1, sdnet.b2]:
        if param.requires_grad:
            total += param.numel()
            zeros += (param == 0).sum().item()
    return 100 * zeros / total if total > 0 else 0

model_stats = {}

params_dict = {
    'A': {
        'init_size': 20,
        'max_size': 150,
        'sparse': True,
        'sparse_ratio': 0.3,
        'loop_num': 10,
        'full_data': False,
        'remove': True,
    },
    'B': {
        'init_size': 100,
        'max_size': 150,
        'sparse': True,
        'sparse_ratio': 0.9,
        'loop_num': 10,
        'full_data': False,
        'remove': True,
    },
    'C': {
        'init_size': 100,
        'max_size': 150,
        'sparse': True,
        'sparse_ratio': 0.9,
        'loop_num': 10,
        'full_data': False,
        'remove': True,
    }
}
# Evaluate each of SCANN's training schemes 
for SCHEME in ["A", "B", "C"]:
    # # skipping this section for now
    # break
    print(f"--- Starting training for Scheme {SCHEME} ---")

    # Set seeds for reproducibility
    RANDOM_STATE = 0
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load scheme-specific parameters
    params = params_dict[SCHEME]

    save_dir = f'/content/drive/MyDrive/test_record_ablation_{SCHEME.lower()}'
    os.makedirs(save_dir, exist_ok=True)

    sdnet = SDNN(in_num, out_num, batch_size=256, scheme=SCHEME, **params)
    sdnet.structureInit(load=False, sparse=params['sparse'], ratio=params['sparse_ratio'])
    sdnet.loadData(mode='train', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    sdnet.train(10, save_dir)

    for i in range(params['loop_num']):
        sdnet.addConnection(mode='grad', percentile={'m2': 70, 'm1': 70, 'm3': 70, 'm4': 70}, full_data=False)
        sdnet.train(10, save_dir)

        if SCHEME == "A":
            sdnet.cellDivision(full_data=params['full_data'])
            sdnet.train(10, save_dir)

        elif SCHEME in ["B", "C"]:
            sdnet.pruneConnections(prune_ratio=0.2)
            sdnet.train(10, save_dir)

    # Save model weights
    final_checkpoint_path = os.path.join(save_dir, 'SDNN_model_best.pth.tar')
    torch.save({
      'epoch': sdnet.epoch,
      'best_acc': sdnet.best_acc,
      'now_acc': sdnet.now_acc,
      'state_dict': {
          'w1': sdnet.w1.data,
          'm1': sdnet.m1.data,
          'w2': sdnet.w2.data,
          'm2': sdnet.m2.data,
          'w3': sdnet.w3.data,
          'm3': sdnet.m3.data,
          'w4': sdnet.w4.data,
          'm4': sdnet.m4.data,
          'b1': sdnet.b1.data,
          'b2': sdnet.b2.data,
      },
      'active_index': sdnet.active_index,
      'optimizer': sdnet.optimizer.state_dict(),
    }, final_checkpoint_path)

    # Record stats
    model_file_size = os.path.getsize(final_checkpoint_path) / (1024 * 1024)
    num_params = sum(p.numel() for p in [sdnet.w1, sdnet.w2, sdnet.w3, sdnet.w4, sdnet.b1, sdnet.b2] if p.requires_grad)
    sparsity = compute_sparsity(sdnet)

    print(f"[SCANN] Training complete for Scheme {SCHEME}")
    print(f"[SCANN] Final model size: {model_file_size:.2f} MB")
    print(f"[SCANN] Trainable parameters: {num_params}")
    print(f"[SCANN] Sparsity: {sparsity:.2f}%")

    model_stats[SCHEME] = {
        'Model Size (MB)': model_file_size,
        'Num Params': num_params,
        'Sparsity (%)': sparsity
    }

import os
import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

checkpoint = torch.load(checkpoint_path)
print(checkpoint.keys())

### Fine-tuning ###

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

# Resplit the original data (real data)
# ---------------------

X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X, y, test_size=0.2, random_state=42)


X_train_real = torch.tensor(X_train_rp_final, dtype=torch.float32)
X_test_real = torch.tensor(X_test_rp_final, dtype=torch.float32)

fine_tune_metrics = {}  # Stores metrics per scheme
scheme_accuracies= {}

for SCHEME in ["A", "B", "C"]:


        # ----- Pre-fine-tuning evaluation -----
    sdnet.loadData(mode='test', X_train=X_train_real, y_train=y_train_real,
                  X_test=X_test_real, y_test=y_test_real)

    total = 0
    correct = 0
    with torch.no_grad():
        for data in sdnet.testloader:
            inputs, labels = data
            outputs = sdnet.forward(inputs, retain_grad=False)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f"[SCANN] Pre-Fine-Tuning Accuracy for Scheme {SCHEME}: {test_accuracy:.4f}")
    scheme_accuracies[SCHEME] = test_accuracy

    print(f"\n[SCANN] Fine-tuning Scheme {SCHEME} on real data")

    params = {
        'A': {'init_size': 20, 'max_size': 150, 'sparse': True, 'sparse_ratio': 0.3,
              'loop_num': 25, 'full_data': False, 'remove': True},
        'B': {'init_size': 100, 'max_size': 150, 'sparse': True, 'sparse_ratio': 0.9,
              'loop_num': 15, 'full_data': False, 'remove': True},
        'C': {'init_size': 100, 'max_size': 150, 'sparse': True, 'sparse_ratio': 0.9,
              'loop_num': 15, 'full_data': False, 'remove': True}
    }[SCHEME]

    checkpoint_path = os.path.join('/content/drive/MyDrive/test_record_ablation_' + SCHEME.lower(), 'SDNN_model_best.pth.tar')
    sdnet = SDNN(in_num, out_num, batch_size=256, scheme=SCHEME, **params)
    sdnet.structureInit(load=True, file=checkpoint_path)

    # Fine-tune
    sdnet.loadData(mode='train', X_train=X_train_real, y_train=y_train_real,
                   X_test=X_test_real, y_test=y_test_real)

    fine_tune_epochs = 20
    save_path = f'real_finetune_{SCHEME.lower()}'
    os.makedirs(save_path, exist_ok=True)
    sdnet.train(fine_tune_epochs, save_path)

    # Evaluate on test set
    sdnet.loadData(mode='test', X_train=X_train_real, y_train=y_train_real,
                   X_test=X_test_real, y_test=y_test_real)

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for data in sdnet.testloader:
            inputs, labels = data
            outputs = sdnet.forward(inputs, retain_grad=False)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs > 0.5).long()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Binary classification metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"\n[{SCHEME}] Fine-Tuned Metrics:")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"AUC          : {auc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    fine_tune_metrics[SCHEME] = {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc,
        'Confusion Matrix': conf_matrix,
        'Labels': all_labels,  
        'Probs': all_probs     
    }
