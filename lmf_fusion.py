from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.rea= nn.ReLU(),
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.model = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LeakyReLU(0.25),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):

        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = self.linear_1(dropped)
        y_2 = self.linear_2(y_1)

        return y_2
class LMF_fusion(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dims, hidden_dims, dropouts, output_dim, rank, use_softmax=False):#text_out
        super(LMF_fusion, self).__init__()
        self.mRNA_in = input_dims[0]
        self.meth_in = input_dims[1]
        self.miRNA_in = input_dims[2]

        self.mRNA_hidden = hidden_dims[0]
        self.meth_hidden = hidden_dims[1]
        self.miRNA_hidden = hidden_dims[2]

        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax

        self.mRNA_prob = dropouts[0]
        self.meth_prob = dropouts[1]
        self.miRNA_prob = dropouts[2]
        self.post_fusion_prob = dropouts[3]

        # define the pre-fusion subnetworks
        self.mRNA_subnet = SubNet(self.mRNA_in, self.mRNA_hidden, self.mRNA_prob)
        self.meth_subnet = SubNet(self.meth_in, self.meth_hidden, self.meth_prob)
        self.miRNA_subnet = SubNet(self.meth_in, self.meth_hidden, self.meth_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)

        self.mRNA_factor = Parameter(torch.Tensor(self.rank, self.mRNA_hidden + 1, self.output_dim))
        self.meth_factor = Parameter(torch.Tensor(self.rank, self.meth_hidden + 1, self.output_dim))
        self.miRNA_factor = Parameter(torch.Tensor(self.rank, self.meth_hidden + 1, self.output_dim))

        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))

        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        xavier_normal(self.mRNA_factor)
        xavier_normal(self.meth_factor)
        xavier_normal(self.miRNA_factor)
        xavier_normal(self.fusion_weights)

        self.fusion_bias.data.fill_(0)
    def analyze_omics_contributions( self,fusion_mRNA, fusion_meth, fusion_miRNA):
        mRNA_contribution = torch.mean(torch.abs(fusion_mRNA)).item()
        meth_contribution = torch.mean(torch.abs(fusion_meth)).item()
        miRNA_contribution = torch.mean(torch.abs(fusion_miRNA)).item()

        total = mRNA_contribution + meth_contribution + miRNA_contribution

        mRNA_ratio = mRNA_contribution / total
        meth_ratio = meth_contribution / total
        miRNA_ratio = miRNA_contribution / total

        plt.figure(figsize=(10, 6))
        labels = ['mRNA', 'Methylation', 'miRNA']
        sizes = [mRNA_ratio, meth_ratio, miRNA_ratio]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Omics Modality Contribution Ratio')
        plt.savefig('omics_contribution_ratio.png', dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(8, 6))
        x_pos = np.arange(len(labels))
        plt.bar(x_pos, sizes, color=colors)
        plt.xlabel('Omics Modality')
        plt.ylabel('Contribution Ratio')
        plt.title('Contribution of Different Omics Modalities')
        plt.xticks(x_pos, labels)
        plt.ylim(0, 1)
        plt.savefig('omics_contribution_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        return {
            'mRNA_ratio': mRNA_ratio,
            'meth_ratio': meth_ratio,
            'miRNA_ratio': miRNA_ratio,
            'absolute_contributions': {
                'mRNA': mRNA_contribution,
                'methylation': meth_contribution,
                'miRNA': miRNA_contribution
            }
        }
    def analyze_rank_wise_contributions(self, fusion_mRNA, fusion_meth, fusion_miRNA,fusion_weights,rank):
        rank_contributions = []
        for i in range(int(rank.item())):
            rank_weight = fusion_weights[0, i].item()
            mRNA_rank_contribution = torch.mean(torch.abs(fusion_mRNA[:, i, :])).item()
            meth_rank_contribution = torch.mean(torch.abs(fusion_meth[:, i, :])).item()
            miRNA_rank_contribution = torch.mean(torch.abs(fusion_miRNA[:, i, :])).item()
            total_rank = mRNA_rank_contribution + meth_rank_contribution + miRNA_rank_contribution
            rank_contributions.append({
                'rank': i,
                'weight': rank_weight,
                'mRNA_ratio': mRNA_rank_contribution / total_rank,
                'meth_ratio': meth_rank_contribution / total_rank,
                'miRNA_ratio': miRNA_rank_contribution / total_rank
            })
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            ranks = [rc['rank'] for rc in rank_contributions]
            weights = [rc['weight'] for rc in rank_contributions]
            axes[0].bar(ranks, weights)
            axes[0].set_xlabel('Rank')
            axes[0].set_ylabel('Weight')
            axes[0].set_title('Fusion Weights for Each Rank')
            mRNA_ratios = [rc['mRNA_ratio'] for rc in rank_contributions]
            meth_ratios = [rc['meth_ratio'] for rc in rank_contributions]
            miRNA_ratios = [rc['miRNA_ratio'] for rc in rank_contributions]
            x = np.arange(len(ranks))
            width = 0.25
            axes[1].bar(x - width, mRNA_ratios, width, label='mRNA')
            axes[1].bar(x, meth_ratios, width, label='Methylation')
            axes[1].bar(x + width, miRNA_ratios, width, label='miRNA')
            axes[1].set_xlabel('Rank')
            axes[1].set_ylabel('Contribution Ratio')
            axes[1].set_title('Modality Contribution by Rank')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(ranks)
            axes[1].legend()
            plt.tight_layout()
            plt.savefig('rank_wise_contributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            return rank_contributions
    def analyze_sample_wise_contributions(self, fusion_mRNA, fusion_meth, fusion_miRNA, batch_size,n_samples=10):
        sample_contributions = []
        for i in range(min(n_samples, batch_size)):
            mRNA_sample_contribution = torch.mean(torch.abs(fusion_mRNA[i])).item()
            meth_sample_contribution = torch.mean(torch.abs(fusion_meth[i])).item()
            miRNA_sample_contribution = torch.mean(torch.abs(fusion_miRNA[i])).item()

            total_sample = mRNA_sample_contribution + meth_sample_contribution + miRNA_sample_contribution

            sample_contributions.append({
                'sample': i,
                'mRNA_ratio': mRNA_sample_contribution / total_sample,
                'meth_ratio': meth_sample_contribution / total_sample,
                'miRNA_ratio': miRNA_sample_contribution / total_sample
            })
        samples = [sc['sample'] for sc in sample_contributions]
        mRNA_ratios = [sc['mRNA_ratio'] for sc in sample_contributions]
        meth_ratios = [sc['meth_ratio'] for sc in sample_contributions]
        miRNA_ratios = [sc['miRNA_ratio'] for sc in sample_contributions]

        plt.figure(figsize=(12, 6))

        plt.plot(samples, mRNA_ratios, 'o-', label='mRNA')
        plt.plot(samples, meth_ratios, 'o-', label='Methylation')
        plt.plot(samples, miRNA_ratios, 'o-', label='miRNA')

        plt.xlabel('Sample Index')
        plt.ylabel('Contribution Ratio')
        plt.title('Modality Contribution Across Samples')

        plt.legend()

        plt.grid(True, alpha=0.3)
        plt.savefig('sample_wise_contributions.png', dpi=300, bbox_inches='tight')
        plt.close()

        return sample_contributions
    def analyze_output_wise_contributions(self, fusion_mRNA, fusion_meth, fusion_miRNA,output_dim):
        output_contributions = []
        for i in range(output_dim):
            # 计算每个输出维度的贡献
            mRNA_output_contribution = torch.mean(torch.abs(fusion_mRNA[:, :, i])).item()
            meth_output_contribution = torch.mean(torch.abs(fusion_meth[:, :, i])).item()
            miRNA_output_contribution = torch.mean(torch.abs(fusion_miRNA[:, :, i])).item()

            total_output = mRNA_output_contribution + meth_output_contribution + miRNA_output_contribution
            output_contributions.append({
                'output': i,
                'mRNA_ratio': mRNA_output_contribution / total_output,
                'meth_ratio': meth_output_contribution / total_output,
                'miRNA_ratio': miRNA_output_contribution / total_output
            })

        outputs = [oc['output'] for oc in output_contributions]
        mRNA_ratios = [oc['mRNA_ratio'] for oc in output_contributions]
        meth_ratios = [oc['meth_ratio'] for oc in output_contributions]
        miRNA_ratios = [oc['miRNA_ratio'] for oc in output_contributions]

        x = np.arange(len(outputs))
        width = 0.25
        plt.figure(figsize=(12, 6))
        plt.bar(x - width, mRNA_ratios, width, label='mRNA')
        plt.bar(x, meth_ratios, width, label='Methylation')
        plt.bar(x + width, miRNA_ratios, width, label='miRNA')
        plt.xlabel('Output Dimension')
        plt.ylabel('Contribution Ratio')
        plt.title('Modality Contribution Across Output Dimensions')
        plt.xticks(x, outputs)
        plt.legend()
        plt.savefig('output_wise_contributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        return output_contributions


    def forward(self, mRNA_x, meth_x, miRNA_x,W=False):
        mRNA_h = self.mRNA_subnet(mRNA_x)
        meth_h = self.meth_subnet(meth_x)
        miRNA_h = self.miRNA_subnet(miRNA_x)
        batch_size = mRNA_h.data.shape[0]
        if mRNA_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _mRNA_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), mRNA_h), dim=1)
        _meth_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), meth_h), dim=1)
        _miRNA_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), miRNA_h), dim=1)

        fusion_mRNA= torch.matmul(_mRNA_h, self.mRNA_factor)
        fusion_meth = torch.matmul(_meth_h, self.meth_factor)
        fusion_miRNA = torch.matmul(_miRNA_h, self.miRNA_factor)
        fusion_zy = fusion_mRNA * fusion_meth * fusion_miRNA
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output)
        if W==True:
            mRNA_importance = torch.norm(self.mRNA_factor, p=2, dim=0)
            meth_importance = torch.norm(self.meth_factor, p=2, dim=0)
            miRNA_importance = torch.norm(self.miRNA_factor, p=2, dim=0)

            mRNA_mean = mRNA_importance.mean().item()
            meth_mean = meth_importance.mean().item()
            miRNA_mean = miRNA_importance.mean().item()

            print(f"mRNA模态重要性: {mRNA_importance.mean().item()}")
            print(f"甲基化模态重要性: {meth_importance.mean().item()}")
            print(f"miRNA模态重要性: {miRNA_importance.mean().item()}")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            modalities = ['mRNA', 'Methylation', 'miRNA']
            importance_values = [mRNA_mean, meth_mean, miRNA_mean]
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            bars = ax1.bar(modalities, importance_values, color=colors)
            ax1.set_title('各模态重要性比较', fontsize=16, fontweight='bold')
            ax1.set_ylabel('平均L2范数值', fontsize=12)
            for bar, value in zip(bars, importance_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            plt.tight_layout()
            save_path = "modality_importance_bar.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            for bar, value in zip(bars, importance_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                         f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            total = sum(importance_values)
            percentages = [v / total * 100 for v in importance_values]
            explode = (0.05, 0.05, 0.05)
            wedges, texts, autotexts = ax2.pie(importance_values, explode=explode, labels=modalities,
                                               colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
            ax2.set_title( fontsize=16, fontweight='bold')
            overall_contributions = self.analyze_omics_contributions( fusion_mRNA, fusion_meth, fusion_miRNA)

            print("Overall Contributions:")
            print(f"mRNA: {overall_contributions['mRNA_ratio']:.3f}")
            print(f"Methylation: {overall_contributions['meth_ratio']:.3f}")
            print(f"miRNA: {overall_contributions['miRNA_ratio']:.3f}")

        return output
