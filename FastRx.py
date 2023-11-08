import os
import time
import dill
import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from prettytable import PrettyTable
from collections import defaultdict
from util import llprint, multi_label_metric, ddi_rate_score, set_seed
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CyclicLR, StepLR

from models import FastRx
from ablations import FastRx_wo_GCN, FastRx_wo_Diag, FastRx_wo_Proc, FastRx_wo_CNN1D

from torch.utils.tensorboard import SummaryWriter

set_seed(1203)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default='FastRx', help="model name")
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--sched', type=str, default='Cyclic')
parser.add_argument('--resume_path', type=str, default='Epoch_72_TARGET_0.06_JA_0.5422_DDI_0.06674.model', help='resume path')
parser.add_argument('--log_dir', type=str, default='', help='tensorboard log dir')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--base_lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--target_ddi', type=float, default=0.06, help='target ddi')
parser.add_argument('--kp', type=float, default=0.05, help='coefficient of P signal')
parser.add_argument('--dim', type=int, default=64, help='dimension')
parser.add_argument('--cuda', type=int, default=0, help='which cuda')

parser.add_argument('--alpha', type=float, default=0.95, help='alpha for loss')
parser.add_argument('--threshold', type=float, default=0.5, help='threshold for prediction')

args = parser.parse_args()
print(args)
save_path = os.path.join('saved', args.model_name, args.log_dir)
print(save_path)

if not args.Test:
    os.makedirs(save_path, exist_ok=True)

    log_dir = 'tensorboard/' + args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir, "comment")

# evaluate
def eval(model, data_eval, voc_size, epoch, testing=False, out_file=None):
    model.eval()

    smm_record = []
    ja, prauc, avg_f1 = [[] for _ in range(3)]
    med_cnt, visit_cnt = 0, 0
    case_study = defaultdict(dict)

    for step, input in enumerate(data_eval):
        if len(input) < 2: continue
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input):

            target_output, _ = model(input[:adm_idx+1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # prediction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, _, _, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        case_study[adm_ja] = {'ja': adm_ja, 'patient': input, 'y_label': y_pred_label}

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path='data/output/ddi_A_final.pkl')
    dill.dump(case_study, open(os.path.join(save_path, 'case_study.pkl'), 'wb'))

    llprint('\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_f1), med_cnt / visit_cnt
    ))
    if out_file is not None:
        out_file.write("DDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_f1), med_cnt / visit_cnt
        ))
    if not testing:
        writer.add_scalar(f"evaluate/ddi", ddi_rate, epoch)
        writer.add_scalar(f"evaluate/med", med_cnt / visit_cnt, epoch)
        writer.add_scalar(f"evaluate/ja", np.mean(ja), epoch)
        writer.add_scalar(f"evaluate/avg_f1", np.mean(avg_f1), epoch)
        writer.add_scalar(f"evaluate/prauc", np.mean(prauc), epoch)

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_f1), med_cnt / visit_cnt, case_study

def test(model, data_test, voc_size, model_path):
    set_seed(97)

    print('\nTesting model...\n')
    tic = time.time()
    result = []
    test_folder = f"tested/{args.model_name}"
    os.makedirs(test_folder, exist_ok=True)
    with open(f"{test_folder}/{model_path}.txt", 'w') as f:
        f.write(str(args) + '\n'*2)
        best_ja = 0

        for _ in range(10):
            test_sample = np.random.choice(data_test, round(len(data_test) * 0.8), replace=True)
            ddi_rate, ja, prauc, avg_f1, avg_med, case_study = eval(model, test_sample, voc_size, 0, testing=True, out_file=f)
            if best_ja < ja:
                best_ja = ja
                with open(f"{test_folder}/{model_path}.pkl", 'wb') as file:
                    dill.dump(case_study, file)

            result.append([ddi_rate, ja, avg_f1, prauc, avg_med])

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        # outstring = ""
        labels = ['DDI', 'JA', 'AVG_F1', 'PRAUC', 'AVG_MED']
        table = PrettyTable()
        table.field_names = ["", "MEAN", "STD"]

        for m, s, l in zip(mean, std, labels):
            table.add_row([l, f'{m:.4f}', f'{s:.4f}'])

        print(str(table) + '\n')
        total_time = time.time() - tic
        print ('test time: {}'.format(total_time))

        f.write(str(table) + '\n')
        f.write('test time: {}'.format(total_time))

def transfer_procedure(data, diag_voc):
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j][1])):
                data[i][j][1][k] += len(diag_voc.idx2word) + 1
    return data

def main():

    # load data
    data_path = 'data/output/records_final.pkl'
    voc_path = 'data/output/voc_final.pkl'

    ehr_adj_path = 'data/output/ehr_adj_final.pkl'
    ddi_adj_path = 'data/output/ddi_A_final.pkl'
    device = torch.device('cuda:{}'.format(args.cuda))

    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    data = dill.load(open(data_path, 'rb'))

    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    data = transfer_procedure(data, diag_voc)

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    if args.model_name == 'FastRx':
        model = FastRx(voc_size, ehr_adj, ddi_adj, emb_dim=args.dim, device=device)
    elif args.model_name == 'FastRx_wo_GCN':
        model = FastRx_wo_GCN(voc_size, ehr_adj, ddi_adj, emb_dim=args.dim, device=device)
    elif args.model_name == 'FastRx_wo_Diag':
        model = FastRx_wo_Diag(voc_size, ehr_adj, ddi_adj, emb_dim=args.dim, device=device)
    elif args.model_name == 'FastRx_wo_Proc':
        model = FastRx_wo_Proc(voc_size, ehr_adj, ddi_adj, emb_dim=args.dim, device=device)
    elif args.model_name == 'FastRx_wo_CNN1D':
        model = FastRx_wo_CNN1D(voc_size, ehr_adj, ddi_adj, emb_dim=args.dim, device=device)
    else:
        raise('Please choose the correct model.')

    if args.Test:
        model.load_state_dict(torch.load(open(f'{save_path}/' + args.resume_path, 'rb')), strict=False)
        model.to(device=device)
        test(model, data_test, voc_size, args.resume_path)
        print(args.log_dir)
        print(args.resume_path)
        return

    model.to(device=device)

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    if args.sched == 'StepLR':
        scheduler = StepLR(optimizer, step_size=25, gamma=0.95)
    elif args.sched == 'Reduce':
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor=0.95)
    elif args.sched == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer,
                                T_max = args.num_epochs,    # Maximum number of iterations.
                                eta_min = 5e-5)             # Minimum learning rate.
    elif args.sched == 'Cyclic':
        scheduler = CyclicLR(optimizer, base_lr=9.75e-5, max_lr=1.25e-4, cycle_momentum=False,
                             step_size_up=10, step_size_down=20, mode='triangular') # triangular, triangular2, exp_range
    # start iterations
    history = defaultdict(list)
    best_epoch, best_ja, best_ddi = 0, 0, 1e9
    best_model_path = ''

    current_iteration = 0

    # copy models file before training
    os.system(f"cp FastRx.py {save_path}")
    os.system(f"cp models.py {save_path}")

    for epoch in range(args.num_epochs):
        tic = time.time()
        print ('\nepoch {} --------------------------'.format(epoch))

        model.train()
        mean_loss = 0
        for step, input in enumerate(data_train):
            loss = 0
            if len(input) < 2: continue

            def criterion(preds):
                result, loss_ddi = preds
                loss_bce = F.binary_cross_entropy_with_logits(result, torch.FloatTensor(loss_bce_target).to(device))
                loss_multi = F.multilabel_margin_loss(F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device))

                result = F.sigmoid(result).detach().cpu().numpy()[0]
                result[result >= args.threshold] = 1
                result[result < args.threshold] = 0
                y_label = np.where(result == 1)[0]
                current_ddi_rate = ddi_rate_score([[y_label]], path='data/output/ddi_A_final.pkl')

                if current_ddi_rate <= args.target_ddi:
                    loss = args.alpha * loss_bce + (1 - args.alpha) * loss_multi
                else:
                    beta = min(0, 1 + (args.target_ddi - current_ddi_rate) / args.kp)
                    loss = beta * (args.alpha * loss_bce + (1 - args.alpha) * loss_multi) + (1 - beta) * loss_ddi
                return loss

            for idx, adm in enumerate(input):
                current_iteration += 1

                seq_input = input[:idx+1]
                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for idx, item in enumerate(adm[2]):
                    loss_multi_target[0][idx] = item

                writer.add_scalar(f"train/lr", optimizer.param_groups[0]['lr'], current_iteration)

                loss = criterion(model(seq_input))

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                writer.add_scalar(f"total_loss", loss.item(), current_iteration)
            mean_loss += loss.item()
            llprint('\rtraining step: {} / {}'.format(step, len(data_train)))

        if args.sched == 'Reduce':
            scheduler.step(mean_loss/len(data_train))
        else:
            scheduler.step()

        print ()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_f1, avg_med, _ = eval(model, data_eval, voc_size, epoch)
        print ('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        if epoch >= 5:
            print ('ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}, lr: {}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['prauc'][-5:]),
                optimizer.param_groups[0]['lr']
                ))
            writer.add_scalar(f"train/ddi", np.mean(history['ddi_rate'][-5:]), epoch)
            writer.add_scalar(f"train/med", np.mean(history['med'][-5:]), epoch)
            writer.add_scalar(f"train/ja", np.mean(history['ja'][-5:]), epoch)
            writer.add_scalar(f"train/avg_f1", np.mean(history['avg_f1'][-5:]), epoch)
            writer.add_scalar(f"train/prauc", np.mean(history['prauc'][-5:]), epoch)

        model_name = 'Epoch_{}_TARGET_{:.2}_JA_{:.4}_DDI_{:.4}.model'.format(epoch, args.target_ddi, ja, ddi_rate)

        if epoch != 0 and best_ja <= ja:
            if best_ja == ja:
                if best_ddi > ddi_rate:
                    best_epoch = epoch
                    best_ja = ja
                    best_ddi = ddi_rate
                    best_model_path = model_name
                    torch.save(model.state_dict(), open(os.path.join(save_path, model_name), 'wb'))
            else:
                best_epoch = epoch
                best_ja = ja
                best_ddi = ddi_rate
                best_model_path = model_name
                torch.save(model.state_dict(), open(os.path.join(save_path, model_name), 'wb'))

        print('best_epoch: {}, best_ja: {:.4f}, best_ddi: {:.4f}'.format(best_epoch, best_ja, best_ddi))
        print(os.path.join(save_path, model_name))

    dill.dump(history, open(os.path.join(save_path, 'history_{}.pkl'.format(args.model_name)), 'wb'))
    print(args.log_dir, os.path.join(save_path, best_model_path))
    os.system(f"mv train.log {save_path}/")

if __name__ == '__main__':
    main()