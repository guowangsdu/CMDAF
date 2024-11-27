import argparse
from utils.en_train import EnConfig, EnRun
from utils.ch_train import ChConfig, ChRun
from distutils.util import strtobool

def main(args):
    if args.dataset != 'sims':
        EnRun(EnConfig(batch_size=args.batch_size,learning_rate=args.lr,seed=args.seed, model=args.model, tasks = args.tasks,
                                    cme_version=args.cme_version, dataset_name=args.dataset,num_hidden_layers=args.num_hidden_layers,
                                    context=args.context, text_context_len=args.text_context_len, audio_context_len=args.audio_context_len,
                                    loss_func=args.loss_func, grad_steps=args.grad_steps,device=args.device,config_id=args.config_id))
    else:
        ChRun(ChConfig(batch_size=args.batch_size,learning_rate=args.lr,seed=args.seed, model=args.model, tasks = args.tasks,
                                    cme_version=args.cme_version, num_hidden_layers=args.num_hidden_layers,
                                   context=args.context, text_context_len=args.text_context_len,audio_context_len=args.audio_context_len,
                                    grad_steps=args.grad_steps,device=args.device,config_id=args.config_id))

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-6,
                        help='learning rate, recommended: 5e-6 for mosi, mosei, 1e-5 for sims')
    parser.add_argument('--model', type=str, default='cme', help='concatenate(cc) or cross-modality encoder(cme)')
    parser.add_argument('--cme_version', type=str, default='v1', help='version')
    parser.add_argument('--dataset', type=str, default='mosi', help='dataset name: mosi, mosei, sims')
    parser.add_argument('--num_hidden_layers', type=int, default=5,
                        help='number of hidden layers for cross-modality encoder')
    parser.add_argument('--tasks', type=str, default='MTA',
                        help='losses to train: M: multi-modal, T: text, A: audio (defalut: MTA))')
    parser.add_argument('--context', default=True, help='incorporate context or not', dest='context',
                        type=lambda x: bool(strtobool(x)))
    parser.add_argument('--text_context_len', type=int, default=2)
    parser.add_argument('--audio_context_len', type=int, default=1)

    parser.add_argument('--loss_func', type=str, default="L1", help='kind of loss function ')
    parser.add_argument('--grad_steps', type=int, default=4, help='gradient accumulation steps')
    parser.add_argument('--device', type=str, default='0', help='id of cuda', choices=['0', '1', '2', '3'])
    parser.add_argument("--config_id", type=str, default='default config', help='the id of the config in this run')

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = create_args()

    main(args)





