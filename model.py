import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        #self.drop_prob = 0.5
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.word_embeddings = nn.Embedding(vocab_size,embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            dropout=0.5, batch_first=True)
        
        #self.dropout = nn.Dropout(self.drop_prob)
        
        self.fc = nn.Linear(hidden_size,vocab_size)
        
        pass
    
    def forward(self, features, captions):
        captions_embed = self.word_embeddings(captions[:,:-1])  #TO remove the end token
        caption = torch.cat((features.unsqueeze(dim = 1),captions_embed),dim = 1)
        lstm_ouput,_ = self.lstm(caption)
        #output = self.dropout(self.fc(lstm_ouput))
        output1 = self.fc(lstm_ouput)
        #output = self.fc(output)
        return output1
        pass

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        hidden = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                  torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
        
        predictions = []
        
        for i in range(max_len):
            lstm_ouput,hidden = self.lstm(inputs,hidden)
            #output = self.dropout(self.fc(lstm_ouput))
            out = self.fc(lstm_ouput)
            out = out.squeeze(1)
            out_word  = out.argmax(dim=1)
            predictions.append(out_word.item())
            
            inputs = self.word_embeddings(out_word.unsqueeze(0))
        return predictions
        pass