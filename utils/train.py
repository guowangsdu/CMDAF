import os
import time

import torch
from torch import nn
from tqdm import tqdm
from metricsTop import MetricsTop
from context_model import rob_d2v_cc_context, rob_d2v_cme_context
from en_model import rob_d2v_cc, rob_d2v_cme
import random
import numpy as np
from data_loader import data_loader
from itertools import chain
import pandas as pd

# global variable
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str


class EnConfig(object):
    """Configuration class to store the configurations of training.
    """
    def __init__(self,
                train_mode = 'regression',
                loss_weights = {
                    'M':1,
                    'T':1,
                    'A':1,
                },
                 model_save_path = '../checkpoint/',
                 learning_rate = 1e-5,
                 epochs = 20,
                 dataset_name = 'mosei',
                 early_stop = 8,
                 seed = 0,
                 dropout=0.3,
                 model='cc',
                 batch_size = 16,
                 multi_task = True,
                 model_size = 'small',
                 cme_version = 'v1',
                 num_hidden_layers = 1,
                 tasks = 'M',   # 'M' or 'MTA',
                 context = True,
                 text_context_len = 2,
                 audio_context_len = 1,
                 grad_steps=2,
                 device=0,
                 config_id='base'
                ):

        self.train_mode = train_mode
        self.loss_weights = loss_weights
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.model_save_path = model_save_path+config_id+'/'
        self.config_save_path = 'results/'
        # 判断路径是否存在，如果不存在则创建
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.exists(self.config_save_path):
            os.makedirs(self.config_save_path)

        self.early_stop = early_stop
        self.seed = seed
        self.dropout = dropout
        self.model = model
        self.batch_size = batch_size
        self.multi_task = multi_task
        self.model_size = model_size
        self.cme_version = cme_version
        self.num_hidden_layers = num_hidden_layers
        self.tasks = tasks
        self.context = context
        self.text_context_len = text_context_len
        self.audio_context_len = audio_context_len

        self.grad_steps = grad_steps
        self.device = device
        self.config_id = config_id

        
        
class EnTrainer():
    def __init__(self, config):
 
        self.config = config
        self.criterion = nn.L1Loss() if config.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(config.train_mode).getMetics(config.dataset_name)
        self.tasks = config.tasks
        self.device = torch.device("cuda:"+config.device if torch.cuda.is_available() else "cpu")
        
    def do_train(self, model, data_loader):    
        model.train()
        model.zero_grad()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        total_loss = 0
        gradient_accumulation_steps = self.config.grad_steps
        # Loop over all batches.         
        for i,batch in enumerate(tqdm(data_loader)):
        # for i,batch in enumerate(data_loader):

            text_inputs = batch["text_tokens"].to(self.device)
            text_mask = batch["text_masks"].to(self.device)
            text_context_inputs = batch["text_context_tokens"].to(self.device)
            text_context_mask = batch["text_context_masks"].to(self.device)
            text_and_context_inputs = batch["text_and_context_tokens"].to(self.device)
            text_and_context_mask = batch["text_and_context_masks"].to(self.device)

            audio_inputs = batch["audio_inputs"].to(self.device)
            audio_mask = batch["audio_masks"].to(self.device)
            audio_context_inputs = batch["audio_context_inputs"].to(self.device)
            audio_context_mask = batch["audio_context_masks"].to(self.device)
            audio_and_context_inputs = batch["audio_and_context_inputs"].to(self.device)
            audio_and_context_masks = batch["audio_and_context_masks"].to(self.device)

            targets = batch["targets"].to(self.device).view(-1, 1)

            if self.config.context:
                if self.config.model == "cme":
                    outputs = model(text_inputs, text_mask, text_context_inputs, text_context_mask, text_and_context_inputs, text_and_context_mask, audio_inputs, audio_mask, audio_context_inputs, audio_context_mask, audio_and_context_inputs, audio_and_context_masks)
                else:
                    outputs = model(text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs, audio_mask, audio_context_inputs, audio_context_mask,)
            else:
                outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)
            
            # Compute the training loss.
            if self.config.multi_task:
                loss = 0.0
                for m in self.tasks:
                    sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets)
                    loss += sub_loss
    #                 train_loss[m] += sub_loss.item()*text_inputs.size(0)
                total_loss += loss.item()*text_inputs.size(0)  
            else:
                loss = self.criterion(outputs['M'], targets)        
                total_loss += loss.item()*text_inputs.size(0)
            loss = loss / gradient_accumulation_steps
            loss.backward()
            if (i+1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()  # To zero out the gradients.
                
        total_loss = round(total_loss / len(data_loader.dataset), 4)
#         print('TRAIN'+" >> loss: ",total_loss)
        return total_loss

    def do_test(self, model, data_loader, mode):
        model.eval()   # Put the model in eval mode.
        if self.config.multi_task:
            y_pred = {'M': [], 'T': [], 'A': []}
            y_true = {'M': [], 'T': [], 'A': []}
            total_loss = 0
            val_loss = {
                'M':0,
                'T':0,
                'A':0
            }
        else:
            y_pred = []
            y_true = []
            total_loss = 0

        with torch.no_grad():
            # for batch in data_loader:
            for batch in tqdm(data_loader):                    # Loop over all batches.
                text_inputs = batch["text_tokens"].to(self.device)
                text_mask = batch["text_masks"].to(self.device)
                text_context_inputs = batch["text_context_tokens"].to(self.device)
                text_context_mask = batch["text_context_masks"].to(self.device)
                text_and_context_inputs = batch["text_and_context_tokens"].to(self.device)
                text_and_context_mask = batch["text_and_context_masks"].to(self.device)

                audio_inputs = batch["audio_inputs"].to(self.device)
                audio_mask = batch["audio_masks"].to(self.device)
                audio_context_inputs = batch["audio_context_inputs"].to(self.device)
                audio_context_mask = batch["audio_context_masks"].to(self.device)
                audio_and_context_inputs = batch["audio_and_context_inputs"].to(self.device)
                audio_and_context_masks = batch["audio_and_context_masks"].to(self.device)

                targets = batch["targets"].to(self.device).view(-1, 1)

                if self.config.context:
                    if self.config.model == "cme":
                        outputs = model(text_inputs, text_mask, text_context_inputs, text_context_mask,
                                        text_and_context_inputs, text_and_context_mask, audio_inputs, audio_mask,
                                        audio_context_inputs, audio_context_mask, audio_and_context_inputs,
                                        audio_and_context_masks)
                    else:
                        outputs = model(text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs,
                                        audio_mask, audio_context_inputs, audio_context_mask, )
                else:
                    outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)
                
                # Compute loss.
                if self.config.multi_task:
                    loss = 0.0         
                    for m in self.tasks:
                        sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets)
                        loss += sub_loss
                        val_loss[m] += sub_loss.item()*text_inputs.size(0)
                    total_loss += loss.item()*text_inputs.size(0)
                    # add predictions
                    for m in self.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(targets.cpu())
                else:
                    loss = self.criterion(outputs['M'], targets)        
                    total_loss += loss.item()*text_inputs.size(0)

                    # add predictions
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(targets.cpu())

        if self.config.multi_task:
            for m in self.tasks:
                val_loss[m] = round(val_loss[m] / len(data_loader.dataset), 4)
            total_loss = round(total_loss / len(data_loader.dataset), 4)
            # print(mode+" >> loss: ",total_loss, "   M_loss: ", val_loss['M'], "  T_loss: ", val_loss['T'], "  A_loss: ", val_loss['A'])

            eval_results = {}
            for m in self.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                results = self.metrics(pred, true)
                print('%s: >> ' %(m) + dict_to_str(results))
                eval_results[m] = results
            eval_results = eval_results[self.tasks[0]]
            eval_results['Loss'] = total_loss 
        else:
            total_loss = round(total_loss / len(data_loader.dataset), 4)
            print(mode+" >> loss: ",total_loss)

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            eval_results = self.metrics(pred, true)
            print('%s: >> ' %('M') + dict_to_str(eval_results))
            eval_results['Loss'] = total_loss
        
        return eval_results

def save_metrics(config, epoch, result):
    d = {'epoch': epoch, 'current time': time.asctime(), 'mode':'RH'}
    d.update(result)
    df = pd.DataFrame(d, index=[0])
    df.to_csv(config.config_save_path + '/' + config.config_id + '-results.csv', mode='a', header=False, index=False)
    # print("epoch:   "+str(epoch)+" results saved！")

def save_metrics2(config, epoch, val_result, test_result):
    d = {'epoch': epoch, 'current time': time.asctime(), 'mode':'val'}
    d.update(val_result)

    modified_dict = {key + '_test': value for key, value in test_result.items()}
    d.update(modified_dict)
    df = pd.DataFrame(d, index=[0])
    if epoch == 1:
        df.to_csv(config.config_save_path+'/'+config.config_id+'-results.csv', mode='a', index=False)
    else:
        df.to_csv(config.config_save_path+'/'+config.config_id+'-results.csv', mode='a', header=False, index=False)

    # print("epoch:   "+str(epoch)+"   results saved！")


def print_config(config):
    # 将参数保存到字典中
    args_dict = vars(config)

    # 转换为 DataFrame
    df = pd.DataFrame.from_dict([args_dict])

    # 保存到 CSV 文件
    df.to_csv(config.config_save_path+'/'+config.config_id+'-results.csv', index=False)
    print("config: saved！")
    return True

def set_seed(seed):
    # torch.set_default_tensor_type('torch.FloatTensor')
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_cuda = True

def EnRun(config):


    print_config(config)
    set_seed(config.seed)

    device = torch.device("cuda:"+config.device if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, val_loader = data_loader(config.batch_size, config.dataset_name,
                                                        text_context_length=config.text_context_len,
                                                        audio_context_length=config.audio_context_len)

    if config.context:
        if config.model == 'cc':
            model = rob_d2v_cc_context(config).to(device)
        else:
            model = rob_d2v_cme_context(config).to(device)
        for param in model.data2vec_model.feature_extractor.parameters():
            param.requires_grad = False
    else:
        if config.model == 'cc':
            model = rob_d2v_cc(config).to(device)
        elif config.model == 'cme':
            model = rob_d2v_cme(config).to(device)
        for param in model.data2vec_model.feature_extractor.parameters():
            param.requires_grad = False

    if torch.cuda.is_available():
        model = model.to(device)
        print('Model---------cuda:' + config.device)


    trainer = EnTrainer(config)

    lowest_eval_loss = 100
    highest_eval_acc = 0
    epoch = 0
    best_epoch = 0
    lowest_eval_loss_epoch = 0
    highest_eval_acc_epoch = 0
    while True:
        print('---------------------EPOCH: ', epoch+1, '--------------------')
        epoch += 1
        trainer.do_train(model, train_loader)
        eval_results = trainer.do_test(model, val_loader,"VAL")
        # save_metrics(config,epoch,eval_results)
    #         test_results = trainer.do_test(model, test_loader,"TEST")
        if eval_results['Loss'] < lowest_eval_loss:
            lowest_eval_loss = eval_results['Loss']
            # torch.save(model.state_dict(), config.model_save_path+'RH_loss.pth')
            lowest_eval_loss_epoch = best_epoch= epoch
        if eval_results['Has0_acc_2'] >= highest_eval_acc:
            highest_eval_acc = eval_results['Has0_acc_2']
            # torch.save(model.state_dict(), config.model_save_path+'RH_acc.pth')
            highest_eval_acc_epoch = epoch
        test_results = trainer.do_test(model, test_loader, "TEST")
        save_metrics2(config, epoch, eval_results, test_results)
        # if epoch - best_epoch >= config.early_stop:
        #     print('Best epoch: >> '+str(best_epoch))
        #     break
        if epoch - lowest_eval_loss_epoch >= config.early_stop and epoch - highest_eval_acc_epoch >= config.early_stop:
            print("Ended-----epoch:"+str(epoch))
            break



    # model.load_state_dict(torch.load(config.model_save_path+'RH_loss.pth'))
    # test_results_loss = trainer.do_test(model, test_loader,"TEST")
    # save_metrics(config, 'RH_loss'+str(lowest_eval_loss_epoch), test_results_loss)
    # print('%s: >> ' %('TEST (lowest val loss) ') + dict_to_str(test_results_loss))
    #
    # model.load_state_dict(torch.load(config.model_save_path+'RH_acc.pth'))
    # test_results_acc = trainer.do_test(model, test_loader,"TEST")
    # save_metrics(config, 'RH_acc'+str(highest_eval_acc_epoch), test_results_acc)
    # print('%s: >> ' %('TEST (highest val acc) ') + dict_to_str(test_results_acc))



