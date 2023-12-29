import argparse
import os
import random

import numpy as np
import torch


# ---------------训练参数------------------
def main(args, seed=0, index=0):
    print(args)
    # ------------------设置环境变量--------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model_variant = args.model_variant
    model_path = args.save_path

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path = os.path.join(model_path,str(model_variant))
    batch_size = args.batch_size
    lr = args.lr
    model_name = 'model_' + str(index) + '.pkl'
    model_name = os.path.join(model_path,model_name)
    log_name = 'log_' + str(index) + '.txt'
    log_path = os.path.join(model_path, log_name)
    log = open(log_path,'w')


    return 1, 2, 3


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':

    # ---------------------------------------------添加模型参数-----------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_list', type=str, required=False, default='0')
    parser.add_argument('--n_epochs', type=int, required=False, default=40)
    parser.add_argument('--batch_size', type=int, required=False, default=8)
    parser.add_argument('--accumulate_step', type=int, required=False, default=1)
    parser.add_argument('--lr', type=float, required=False, default=1e-5)
    parser.add_argument('--weight_decay', type=float, required=False, default=0.01)
    parser.add_argument('--model_size', type=str, required=False, default='base')
    parser.add_argument('--model_variant', type=str, required=False, default='csk')
    parser.add_argument('--valid_shuffle', action='store_true', default='是否对验证集和测试集进行洗牌')
    parser.add_argument('--warmup_rate', type=float, required=False, default=0.1)
    parser.add_argument('--add_emotion', action='store_true')
    parser.add_argument('--emotion_dim', type=int, required=False, default=0)
    parser.add_argument('--dag_dropout', type=float, required=False, default=0.1)
    parser.add_argument('--pooler_type', type=str, required=False, default='all')
    parser.add_argument('--window', type=int, required=False, default=10)
    parser.add_argument('--csk_window', type=int, required=False, default=10)
    parser.add_argument('--multigpu', action='store_true')  # 是否使用多GPU进行训练，使用store_true
    parser.add_argument('--trm_dropout', required=False, type=float, default=0.1)
    parser.add_argument('--att_dropout', required=False, type=float, default=0.1)
    parser.add_argument('--nhead', required=False, type=int, default=6)
    parser.add_argument('--ff_dim', required=False, type=int, default=600)  # 默认前馈神经网络的维度
    parser.add_argument('--max_len', required=False, type=int, default=200)
    parser.add_argument('--pe_type', required=False, type=str, default='abs')  # 指定位置编码的类型
    parser.add_argument('--mapping_type', type=str, required=False, default='max')  # 指定映射类型的类型
    parser.add_argument('--utter_dim', type=int, required=False, default=300)  # 指定言语向量的维度
    parser.add_argument('--num_layer', type=int, required=False, default=1)
    parser.add_argument('--conv_encoder', type=str, required=False, default='none')
    parser.add_argument('--rnn_dropout', type=float, required=False, default=0.5)
    parser.add_argument('--seed', nargs='+', type=int, required=False, default=[0, 1, 2, 3, 4])  # 指定随机种子数
    parser.add_argument('--index', nargs='+', type=int, required=False, default=[1, 2, 3, 4, 5])  # 指定索引列表
    parser.add_argument('--save_dir', type=str, required=False, default='saves')

    # -------------------------------之后的打印数据--------------------------------------------------------------

    args_for_main = parser.parse_args()
    seed_list = args_for_main.seed
    index_list = args_for_main.index
    dev_fscore_list = []
    test_fscore_list = []
    model_dir = ''

    # ------------------循环测出dev和test的f1分数----------------
    for sd, idx in zip(seed_list, index_list):
        dev_f1, test_f1, model_dir = main(args_for_main, sd, idx)
        dev_fscore_list.append(dev_f1)
        test_fscore_list.append(test_f1)
        print('sd,idx：', sd, idx)

    # ----------------------计算标准差和平均值-------------------
    dev_fscore_mean = np.round(np.mean(dev_fscore_list) * 100, 2)
    dev_fscore_std = np.round(np.std(dev_fscore_list) * 100, 2)
    test_fscore_mean = np.round(np.mean(test_fscore_list) * 100, 2)
    test_fscore_std = np.round(np.std(test_fscore_list) * 100, 2)
    print('dev_fscore_mean:', dev_fscore_mean)
    print('dev_fscore_std:', dev_fscore_std)
    print('test_fscore_mean:', test_fscore_mean)
    print('test_fscore_std:', test_fscore_std)
    # ----------------------定制目录----------------------------
    logs_path = str(model_dir) + '/log_metrics_' + str(index_list[0]) + '-' + str(index_list[-1]) + '.txt'
    logs = open(logs_path, 'w')
    logs.write(str(args_for_main) + '\n\n')
    log_lines = f'dev fscore: {dev_fscore_mean}(+-{dev_fscore_std})'
    print(log_lines)
    logs.write(log_lines + '\n')
    log_lines = f'test fscore: {test_fscore_mean}(+-{test_fscore_std})'
    print(log_lines)
    logs.write(log_lines + '\n')
    logs.close()
