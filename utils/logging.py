import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_writer(output_directory, log_directory):
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
        
    logging_path=f'{output_directory}/{log_directory}'
    if not os.path.exists(logging_path):
        os.mkdir(logging_path)
        
    writer = SummaryWriter(logging_path)
    return writer

def plot_image(spec, alignments):
    alignments = [ a for a in alignments if a is not None ]
    plots, axes = plt.subplots(len(alignments)+1, 1, figsize=(5*(len(alignments)+1),20))
    axes[0].imshow(spec.detach().cpu().numpy(), origin='lower', aspect='auto')
    for i in range(len(alignments)):
        axes[i+1].imshow(alignments[i].detach().cpu().numpy(), origin='lower', aspect='auto')
    return plots

