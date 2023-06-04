import pylab
import numpy as np
from tqdm import tqdm
import copy
import time
import gc
import xlrd
import xlwt
import torch
import transformers
import sklearn as skl
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from datasets import load_dataset



from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, pipeline

LabtoNum = {'восторг' : 0 , 'восхищение' : 1, 'гнев' : 2, 'горе' : 3,
       'изумление' : 4, 'настороженность' : 5, 'отвращение' : 6, 'ужас' : 7, 'я_none' : 8}

NumToLab = { 0 : 'восторг', 1 : 'восхищение', 2 : 'гнев', 3 : 'горе', 4 : 'изумление', 5 : 'настороженность',
       6 : 'отвращение', 7 : 'ужас', 8 : 'я_none'}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def bootstrap(dataset, subset_len):
    datavals = []
    datalabels = []
    counts = dict()
    
    for i in range(subset_len):
        pos = int(len(dataset)*np.random.random())
        
        counts[dataset[pos][1]] = counts.setdefault(dataset[pos][1],0) + 1
        datavals.append(dataset[pos][0])
        datalabels.append(dataset[pos][1])
        
    print(counts)
    return MyDataset(datavals,datalabels)

class emotion_detector:
    #по сути это должен быть стекинг, только вместо набора из классификаторов тут набор из классифицирующих голов,
    #а берт всегда один.
    #возможно из-за этого будет сложно тренировать весь ансамбль одновременно
    
    def __init__(self, nclasses, nclassifiers, bert = None, tokenizer = None, bootstrap = True,
                 part = 0.9, subsize = 70):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nclasses = nclasses
        self.nclassifiers = nclassifiers
        self.bootstrap = bootstrap
        self.subsize = subsize
        self.part = part
        
        if(bert): #классификатор на основе bert
            self.bert = bert
        else:
            self.bert = BertForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased-conversational', 
                                                                      num_labels=6)

        self.bert.load_state_dict(torch.load('models/DP_bert_cedr_092'))
        
        self.bert.config.id2label = NumToLab
        self.bert.config.label2id = LabtoNum
        
        #сам берт делается необучаемым
        for param in self.bert.bert.parameters():
            param.requires_grad = False 
        self.bert.eval()
        
        if(tokenizer): #токенизатор для bert
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-conversational', 
                                                           do_lower_case=False)
            
        self.pipe = pipeline("text-classification", model=self.bert,tokenizer=self.tokenizer,
                                                device = self.device)
        
        self.heads = [] # здесь хранятся классифицирующие головы 
        for i in range(nclassifiers):
            self.heads.append(torch.nn.Sequential(torch.nn.Linear(768,nclasses)
                                                 ,torch.nn.Softmax(dim = -1)
                                                 ).to(self.device))
        
        self.pipe.model.classifier = copy.deepcopy(self.heads[0])
        
        self.metamodel = torch.nn.Sequential(torch.nn.Linear(nclasses*nclassifiers,nclasses),
                                             torch.nn.Softmax(dim = -1)).to(self.device)
                       #она должна получать на вход список выходных массивов базовых классификаторов и делать из 
                       #него один итоговый список. мб просто принимать классы и выдавать ответ, но это мало инфо
        self.analog = torch.nn.Sequential(torch.nn.Linear(nclassifiers,nclasses),
                                         torch.nn.Softmax(dim = -1)).to(self.device)
        
    def __call__(self,obj):            
        results = self._scores(obj)
        return self.pipe.model.config.id2label[np.argmax(self.metamodel(results.to(self.device)).cpu().detach().numpy())]
        
    def analog(self,obj):
        results = self.results(obj)
        return self.pipe.model.config.id2label[np.argmax(self.analog(results.to(self.device)).cpu().detach().numpy())]
    
    
    def hard_vote(self,obj):
        results = np.zeros(self.nclasses)
        for i in range(self.nclassifiers):
            self.set_head(i)
            
            inp = self.pipe.tokenizer(obj, return_tensors = "pt").to(self.device)
            outputs = self.pipe.model(**inp)
            res = np.argmax(outputs.logits[0].tolist())
            
            results[res] += 1
        
        return self.bert.config.id2label[int(np.argmax(results))]
    
    def soft_vote(self,obj):
        results = np.zeros(self.nclasses)
        for i in range(self.nclassifiers):
            self.set_head(i)
            
            inp = self.pipe.tokenizer(obj, return_tensors = "pt").to(self.device)
            outputs = self.pipe.model(**inp)
            res = np.argmax(outputs.logits[0].tolist())
            
            results[res] += outputs.logits[0].tolist()[res]

        return self.bert.config.id2label[int(np.argmax(results))]
        
    
    def scores(self,obj):
        results = []
        for i in range(self.nclassifiers):
            self.set_head(i)
            
            inp = self.pipe.tokenizer(obj, return_tensors = "pt").to(self.device)
            outputs = self.pipe.model(**inp)
            res = outputs.logits[0].tolist()
            
            results += res
            
        return self.metamodel(torch.tensor(results).to(self.device)).cpu().detach().numpy()
        
    def _scores(self,obj):
        results = []
        for i in range(self.nclassifiers):
            self.set_head(i)
            
            inp = self.pipe.tokenizer(obj, return_tensors = "pt").to(self.device)
            outputs = self.pipe.model(**inp)
            res = outputs.logits[0].tolist()
            
            results += res
            
        return torch.tensor(results).to(self.device)
    
    def _results(self,obj):
        results = []
        for i in range(self.nclassifiers):
            self.set_head(i)
            
            inp = self.pipe.tokenizer(obj, return_tensors = "pt").to(self.device)
            outputs = self.pipe.model(**inp)
            res = np.argmax(outputs.logits[0].tolist())
            
            results.append(res)
            
        return torch.tensor(results, dtype = torch.float32).to(self.device)
        
    def set_head(self,head_n):
        self.pipe.model.classifier = copy.deepcopy(self.heads[head_n])
        
    def save_head(self,head_n):
        self.heads[head_n] = copy.deepcopy(self.pipe.model.classifier)
        
        for param in self.heads[head_n].parameters():
            param.requires_grad = False
            
    def save_(self,fname):
        torch.save([self.pipe.model.bert.state_dict(), 
                    [h.state_dict() for h in self.heads], 
                    self.metamodel.state_dict()], fname)
    
    def load_(self,fname):
        weights = torch.load(fname)
        self.pipe.model.bert.load_state_dict(weights[0])
        for i in range(len(self.heads)):
            self.heads[i].load_state_dict(weights[1][i])
        self.metamodel.load_state_dict(weights[2])           
        
    
    def train(self,train_dataset,val_dataset, n_epochs, n_epochs_weak = None, use_dropout = True):
        
        if(n_epochs_weak == None):
            n_epochs_weak = n_epochs
        
        # 1. разбить обучающаю выборку 
        train_subsets = []
        for i in range(self.nclassifiers):
            if(self.bootstrap == True):
                train_subsets.append(bootstrap(train_dataset,len(train_dataset)))
            elif(self.bootstrap == 'subset'):
                train_subsets.append(ballanced_subset(train_dataset,1.0,self.subsize))
            elif(self.bootstrap == 'bbs'):
                train_subsets.append(keep_ratio_bootstrap(train_dataset,len(train_dataset) / self.nclasses))   
            else:
                train_subsets.append(skl.utils.shuffle(train_dataset))
        # 2. по очереди обучить каждую голову
        traces = []
        valloader = torch.utils.data.DataLoader(val_dataset)
        
        for i in range(self.nclassifiers):
            self.set_head(i)
            #устанавливаем соответствующую голову и создаем загрузчик подвыборки
            trainloader = torch.utils.data.DataLoader(train_subsets[i])
            #для каждой подвыборки считаем веса
            wts = torch.tensor(class_weight.compute_class_weight(class_weight = 'balanced', 
                                             classes = np.unique(np.array(train_subsets[i])[:,1]), 
                                             y = np.array(train_subsets[i])[:,1]), dtype = torch.float32)

            criterion = torch.nn.CrossEntropyLoss(wts.to(self.device))
            optimizer = torch.optim.Adadelta(self.pipe.model.parameters(), lr=1, weight_decay = 0)
            
            self.pipe, tracing = train_model(self.pipe,trainloader,valloader,
                                             criterion,optimizer, num_epochs = n_epochs_weak, 
                                             verbal = False, bert_dropout = use_dropout)
            
            self.save_head(i)
            traces.append(tracing)
            
            
        if(self.bootstrap == True):
            train_dataset_ = bootstrap(train_dataset,len(train_dataset))
        elif(self.bootstrap == 'subset'):
            train_dataset_ = ballanced_subset(train_dataset,1.0,self.subsize)
        elif(self.bootstrap == 'bbs'):
            train_dataset_ = keep_ratio_bootstrap(train_dataset,len(train_dataset) / self.nclasses)
        else:
            train_dataset_ = skl.utils.shuffle(train_dataset)
        
        meta_dataset = MyDataset([self._scores(elm[0]).detach() for elm in train_dataset_], 
                                 torch.tensor([self.pipe.model.config.label2id[elm[1]] for elm in train_dataset_]))
        
        trainloader = torch.utils.data.DataLoader(meta_dataset)
        meta_valset = MyDataset([self._scores(elm[0]).detach() for elm in val_dataset], 
                                torch.tensor([self.pipe.model.config.label2id[elm[1]] for elm in val_dataset]))
        
        valloader = torch.utils.data.DataLoader(meta_valset)
        wts = torch.tensor(class_weight.compute_class_weight(class_weight = 'balanced', 
                                             classes = np.unique(np.array(train_dataset_)[:,1]), 
                                             y = np.array(train_dataset_)[:,1]), dtype = torch.float32)
        criterion = torch.nn.CrossEntropyLoss(wts.to(self.device))
        optimizer = torch.optim.Adadelta(self.metamodel.parameters(), lr=1, weight_decay = 0)
        
        self.metamodel, trace = train_basic(self.metamodel, trainloader, valloader,
                                           criterion,optimizer,num_epochs = n_epochs, verbal = False)
        
        return traces, trace
    
    def train_metaalgo(self, train_dataset, val_dataset, n_epochs):
        
        if(self.bootstrap == True):
            train_dataset_ = bootstrap(train_dataset,len(train_dataset))
        elif(self.bootstrap == 'subset'):
            train_dataset_ = ballanced_subset(train_dataset,1.0,self.subsize)
        elif(self.bootstrap == 'bbs'):
            train_dataset_ = keep_ratio_bootstrap(train_dataset,len(train_dataset) / self.nclasses)
        else:
            train_dataset_ = skl.utils.shuffle(train_dataset)
        
        meta_dataset = MyDataset([self._scores(elm[0]).detach() for elm in train_dataset_], 
                                 torch.tensor([self.pipe.model.config.label2id[elm[1]] for elm in train_dataset_]))
        
        trainloader = torch.utils.data.DataLoader(meta_dataset)
        meta_valset = MyDataset([self._scores(elm[0]).detach() for elm in val_dataset], 
                                torch.tensor([self.pipe.model.config.label2id[elm[1]] for elm in val_dataset]))
        
        #print([i for i in meta_valset])
        
        valloader = torch.utils.data.DataLoader(meta_valset)
        wts = torch.tensor(class_weight.compute_class_weight(class_weight = 'balanced', 
                                             classes = np.unique(np.array(train_dataset_)[:,1]), 
                                             y = np.array(train_dataset_)[:,1]), dtype = torch.float32)
        criterion = torch.nn.CrossEntropyLoss(wts.to(self.device))
        optimizer = torch.optim.Adadelta(self.metamodel.parameters(), lr=1, weight_decay = 0)
        
        self.metamodel, trace = train_basic(self.metamodel, trainloader, valloader,
                                           criterion,optimizer,num_epochs = n_epochs, verbal = False)
        
        return trace
    
    
    def train_analog(self, train_dataset, val_dataset, n_epochs):
        
        if(self.bootstrap == True):
            train_dataset_ = bootstrap(train_dataset,len(train_dataset))
        elif(self.bootstrap == 'subset'):
            train_dataset_ = ballanced_subset(train_dataset,1.0,self.subsize)
        elif(self.bootstrap == 'bbs'):
            train_dataset_ = keep_ratio_bootstrap(train_dataset,len(train_dataset) / self.nclasses)
        else:
            train_dataset_ = skl.utils.shuffle(train_dataset)
        
        meta_dataset = MyDataset([self._results(elm[0]).detach() for elm in train_dataset_], 
                                 torch.tensor([self.pipe.model.config.label2id[elm[1]] for elm in train_dataset_]))
        
        trainloader = torch.utils.data.DataLoader(meta_dataset)
        meta_valset = MyDataset([self._results(elm[0]).detach() for elm in val_dataset], 
                                torch.tensor([self.pipe.model.config.label2id[elm[1]] for elm in val_dataset]))
        
        #print([i for i in meta_valset])
        
        valloader = torch.utils.data.DataLoader(meta_valset)
        wts = torch.tensor(class_weight.compute_class_weight(class_weight = 'balanced', 
                                             classes = np.unique(np.array(train_dataset_)[:,1]), 
                                             y = np.array(train_dataset_)[:,1]), dtype = torch.float32)
        criterion = torch.nn.CrossEntropyLoss(wts.to(self.device))
        optimizer = torch.optim.Adadelta(self.analog.parameters(), lr=1, weight_decay = 0)
        
        self.analog, trace = train_basic(self.analog, trainloader, valloader,
                                           criterion,optimizer,num_epochs = n_epochs, verbal = False)
        
        return trace
        
        
