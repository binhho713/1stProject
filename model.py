"""
Main model (FedX) class representing backbone network and projection heads

"""

import torch.nn as nn

from encoder import resnet18


class ModelFedX(nn.Module):
    def __init__(self, base_model, out_dim, net_configs=None):
        super(ModelFedX, self).__init__()

        if base_model == "resnet18-seperator":
            basemodel = resnet18(predictor=False)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            self.num_ftrs = basemodel.fc.in_features

        elif base_model == "resnet18-predictor":
            basemodel = resnet18(predictor=True)
            self.features = nn.Sequential(*list(basemodel.children())[:-4])
            self.num_ftrs = basemodel.fc.in_features

        else:
            raise ("Invalid model type. Check the config file and pass one of: resnet18 or resnet50")

        self.projectionMLP = nn.Sequential(
            nn.Linear(self.num_ftrs, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

        self.predictionMLP = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)

        # h = h.reshape(x.size(0), -1)
        h.view(-1, self.num_ftrs)
        h = h.squeeze()

        proj = self.projectionMLP(h)
        pred = self.predictionMLP(proj)

        return h, proj, pred

class ModelProtoFL(nn.Module):
    def __init__(self, base_model, out_dim, net_configs=None):
        super(ModelFedX, self).__init__()

        if base_model == "resnet18-seperator":
            basemodel = resnet18(predictor=False)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            self.num_ftrs = basemodel.fc.in_features

        elif base_model == "resnet18-predictor":
            basemodel = resnet18(predictor=True)
            self.features = nn.Sequential(*list(basemodel.children())[:-4])
            self.num_ftrs = basemodel.fc.in_features

        else:
            raise ("Invalid model type. Check the config file and pass one of: resnet18 or resnet50")

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)

        # h = h.reshape(x.size(0), -1)
        h.view(-1, self.num_ftrs)
        h = h.squeeze()

        return h

class LinearMapping_predictor(nn.Module):
    """Prediction head class for linear evaluation"""

    def __init__(self, dim_input, num_class = 10):
        super(LinearMapping_predictor, self).__init__()
        self.num_class = num_class
        self.fc = nn.Linear(dim_input, dim_input, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_input, num_class, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = self.fc1(x)
        out = self.softmax(x)

        return out

class LinearMapping_seperator(nn.Module):
    """Prediction head class for linear evaluation"""

    def __init__(self, dim_input, num_class = 10):
        super(LinearMapping_seperator, self).__init__()
        self.num_class = num_class
        self.fc = nn.Linear(dim_input, dim_input, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_input, 100 * num_class)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc(x))
        x = self.fc1(x)
        out = x.view(-1, 1, self.num_class, 100)

        return out

def init_nets(net_configs, n_parties, args, device="cpu"):
    nets = {net_i: {'predictor':None, 'seperator':None} for net_i in range(n_parties)}
    for net_i in range(n_parties):
        pred = ModelFedX(args.model + '-predictor', args.out_dim, net_configs)
        sep = ModelFedX(args.model + '-seperator', args.out_dim, net_configs)
        pred = pred.cuda()
        sep = sep.cuda()
        nets[net_i]['predictor'] = pred
        nets[net_i]['seperator'] = sep

    return nets

def init_mappingNet(n_parties, args, device="cpu"):
    mNets = {net_i: {'predictor':None, 'seperator':None} for net_i in range(n_parties)}
    for net_i in range(n_parties):
        pred = LinearMapping_predictor(args.out_dim, num_class=10)
        sep = LinearMapping_seperator(args.out_dim, num_class=10)
        pred = pred.cuda()
        sep = sep.cuda()
        mNets[net_i]['predictor'] = pred
        mNets[net_i]['seperator'] = sep

    return mNets

def init_proto_nets(net_configs, n_parties, args, device="cpu"):
    nets = {net_i: {'predictor':None, 'seperator':None} for net_i in range(n_parties)}
    for net_i in range(n_parties):
        pred = ModelProtoFL(args.model + '-predictor', args.out_dim, net_configs)
        sep = ModelProtoFL(args.model + '-seperator', args.out_dim, net_configs)
        pred = pred.cuda()
        sep = sep.cuda()
        nets[net_i]['predictor'] = pred
        nets[net_i]['seperator'] = sep

    return nets

def init_proto_mappingNet(n_parties, args, device="cpu"):
    mNets = {net_i: {'predictor':None, 'seperator':None} for net_i in range(n_parties)}
    for net_i in range(n_parties):
        pred = LinearMapping_predictor(args.out_dim, num_class=10)
        sep = LinearMapping_seperator(args.out_dim, num_class=10)
        pred = pred.cuda()
        sep = sep.cuda()
        mNets[net_i]['predictor'] = pred
        mNets[net_i]['seperator'] = sep

    return mNets