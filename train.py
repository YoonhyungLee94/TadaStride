import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ResNet
import hparams as hp
from utils import *
from dataset import *
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

def validate(model, val_loader, iteration, writer=None):
    model.eval()
    with torch.no_grad():
        n_data, n_correct, val_nll_loss, val_score_loss = 0, 0, 0, 0
        for i, batch in enumerate(val_loader):
            n_data += len(batch[0])
            spec_padded, spec_lengths, labels = [ x.cuda(non_blocking=True) for x in batch]
            output, alignments, score_loss = model(spec_padded, spec_lengths)
            nll_loss = F.nll_loss(output, labels)
            n_correct += (torch.argmax(output, dim=-1)==labels).sum().item()

            val_nll_loss += nll_loss.item() * len(batch[0])
            val_score_loss += score_loss.item() * len(batch[0])

        val_nll_loss /= n_data
        val_score_loss /= n_data

    val_acc = n_correct/n_data
    
    if writer is not None:
        writer.add_scalar('losses_val/nll_loss', val_nll_loss, global_step=iteration)
        writer.add_scalar('losses_val/score_loss', val_score_loss, global_step=iteration)
        writer.add_scalar('losses_val/accuracy', val_acc, global_step=iteration)

        plots = plot_image(spec_padded[-1, :, :spec_lengths[-1]], alignments)
        writer.add_figure('val/plots', plots, global_step=iteration)

    model.train()
    
    return val_acc


def main(hp, args):
    train_dataset = IEMOCAPSet(data_path='./Dataset/IEMOCAP/melspectrogram', subset='train')
    val_dataset = IEMOCAPSet(data_path='./Dataset/IEMOCAP/melspectrogram', subset='validation')
    test_dataset = IEMOCAPSet(data_path='./Dataset/IEMOCAP/melspectrogram', subset='test')

    train_loader = DataLoader(train_dataset,
                              num_workers=4,
                              pin_memory=True,
                              shuffle=True,
                              batch_size=hp.batch_size,
                              drop_last=True,
                              collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset,
                            num_workers=4,
                            pin_memory=True,
                            batch_size=hp.batch_size,
                            collate_fn=collate_fn)
    
    test_loader = DataLoader(test_dataset,
                             num_workers=4,
                             pin_memory=True,
                             batch_size=hp.batch_size,
                             collate_fn=collate_fn)
    
    model = ResNet(hp).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
    lr_warmup_step = len(train_loader) * 10 # first 10 epochs
    writer = get_writer(hp.output_directory, args.logdir)

    iteration = 0
    model.train()
    print(f"Training Start!!! ({count_parameters(model)})")
    for epoch in range(1, hp.epochs+1, 1):
        for i, batch in enumerate(train_loader):
            spec_padded, spec_lengths, labels = [ x.cuda(non_blocking=True) for x in batch ]
            output, alignments, score_loss = model(spec_padded, spec_lengths)
            nll_loss = F.nll_loss(output, labels)
            loss = nll_loss + score_loss
            
            writer.add_scalar('losses_train/nll_loss', nll_loss, global_step=iteration)
            writer.add_scalar('losses_train/score_loss', score_loss, global_step=iteration)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            iteration += 1
            if epoch<=10:
                optimizer.param_groups[0]['lr'] = hp.learning_rate * iteration / lr_warmup_step
            optimizer.step()

        if epoch>10:
            optimizer.param_groups[0]['lr'] = 0.95 * optimizer.param_groups[0]['lr']
        
        plots = plot_image(spec_padded[-1, :, :spec_lengths[-1]], alignments)
        writer.add_figure('train/plots', plots, global_step=iteration)

        val_acc = validate(model, val_loader, iteration, writer)
        
    print(f"Training Finish!!!\n")
    
    last_acc = validate(model, test_loader, iteration)
    writer.add_scalar('TEST ACCURACY', last_acc, global_step=0)
    print(f"TEST ACCURACY: {last_acc}")
    time.sleep(3)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--logdir', type=str, default='TadaStride')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(hp, args)
