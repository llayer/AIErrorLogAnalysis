import numpy as np
from scipy import interp
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import pandas as pd

def plot_skopt_results(skopt_files, skopt_titles):
    
    files = []
    for file in skopt_files:
        skopt = pd.read_hdf(file)
        skopt = skopt.sort_values(by='call', ascending=True).iloc[0:30]
        files.append( skopt )
        
    colors = 'rgbycm'
    
    fig, ax = plt.subplots(figsize=(12,8))
    for counter, opt in enumerate(files):
        #plt.scatter(opt.call, opt.cv_score , c=colors[counter], label = skopt_titles[counter])
        plt.errorbar(list(opt.call), list(opt.cv_score) , yerr=list(opt['std']) , fmt = 'o', label = skopt_titles[counter])
    plt.legend(loc='lower right')
    plt.ylabel('CV(ROC)')
    plt.xlabel('skopt call')


def plot_metric( results, title = 'metric' ):
    c = 'brgcy'
    for counter, result in enumerate(results):
        plt.plot(result.reports, label = counter, c = c[counter])
    plt.xlabel('epochs')
    plt.ylabel(title)


def plot_cv_roc_curve( true_pos, false_pos ):
        
    
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    for i in range(len(true_pos)):
        
        tpr, fpr = true_pos[i], false_pos[i]
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        tprs.append(interp(mean_fpr, fpr, tpr))
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.5,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    
    
    
def compare_roc( rates, titles ):
        
    def plot_roc( true_pos, false_pos, title, color ):
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        for i in range(len(true_pos)):

            tpr, fpr = true_pos[i], false_pos[i]
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            tprs.append(interp(mean_fpr, fpr, tpr))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color=color,
                 label= title + r' (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.5)

    colors = 'bgm'
    for i in range(len(rates)):
        
        tps, fps = rates[i]
        plot_roc(tps, fps, titles[i], colors[i])
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()