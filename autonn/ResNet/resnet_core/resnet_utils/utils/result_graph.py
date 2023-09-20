import pandas as pd
from matplotlib import pyplot as plt


def call_csv(path):
    # path = "./result.csv"
    data_frame = pd.read_csv(path)
    data_frame = data_frame[['model', 'optimizer', 'initial_lr', 'initial_momentum',
                             'epochs', 'train acc', 'val acc', 'test acc', 'time']]
    return data_frame


def split_by_optim(df):
    idx_adam = df[(df['optimizer'] != 'Adam')].index
    df_adam = df.drop(idx_adam)
    df_adam = df_adam.drop(['initial_momentum'], axis=1)

    idx_sgd = df[df['optimizer']!='SGD'].index
    df_sgd = df.drop(idx_sgd)

    idx_nag = df[df['optimizer']!='NAG'].index
    df_nag = df.drop(idx_nag)

    return df_adam, df_sgd, df_nag


def mean_graph_with3(df, lr_list):
    lr1 = df['initial_lr'] == lr_list[0]
    lr2 = df['initial_lr'] == lr_list[1]
    lr3 = df['initial_lr'] == lr_list[2]

    v1 = df[lr1].groupby(['epochs'])[['test acc']].mean()
    v2 = df[lr2].groupby(['epochs'])[['test acc']].mean()
    v3 = df[lr3].groupby(['epochs'])[['test acc']].mean()

    fig = plt.figure()

    ax2 = fig.add_subplot(1, 1, 1)
    ax2.plot(v1, 'k', label='{}'.format(lr_list[0]), marker='o')
    ax2.plot(v2, 'b', label='{}'.format(lr_list[1]), marker='^', linestyle='--')
    ax2.plot(v3, 'r', label='{}'.format(lr_list[2]), marker='s', linestyle=':')
    ax2.set_xticks([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    ax2.set_yticks([0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94])
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('acc')
    ax2.legend(loc='best')
    ax2.grid(axis='y')

    fig.savefig('./result_mean_graph.png')

    return fig


def var_graph_with3(df, lr_list):
    lr1 = df['initial_lr'] == lr_list[0]
    lr2 = df['initial_lr'] == lr_list[1]
    lr3 = df['initial_lr'] == lr_list[2]

    v1 = df[lr1].groupby(['epochs'])[['test acc']].var()
    v2 = df[lr2].groupby(['epochs'])[['test acc']].var()
    v3 = df[lr3].groupby(['epochs'])[['test acc']].var()

    fig = plt.figure()

    ax2 = fig.add_subplot(1, 1, 1)
    ax2.plot(v1, 'k', label='{}'.format(lr_list[0]), marker='o')
    ax2.plot(v2, 'b', label='{}'.format(lr_list[1]), marker='^', linestyle='--')
    ax2.plot(v3, 'r', label='{}'.format(lr_list[2]), marker='s', linestyle=':')
    ax2.set_xticks([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('acc')
    ax2.legend(loc='best')
    ax2.grid(axis='y')

    fig.savefig('./result_var_graph.png')

    return fig, ax2


def more_plot(fig, ax, df, option, label):
    if option == 'mean':
        v = df.groupby(['epochs'])[['test acc']].mean()
    else:
        v = df.groupby(['epochs'])[['test acc']].var()

    ax.plot(v, 'm', label=label, marker='*', linestyle='-.')

    fig.savefig('./result_graph.png')