# 导入未来特性模块，确保与新版本的Python兼容
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入命令行参数解析库
import argparse

# 导入自定义的模型类
from models.cnn_svm import CNN
from models.gru_svm import GruSvm
from models.mlp_svm import MLP

# 导入NumPy库，用于数学运算
import numpy as np

# 导入数据集分割工具
from sklearn.model_selection import train_test_split

# 导入数据加载和预处理工具
from utils.data import load_data
from utils.data import one_hot_encode

# 设置训练参数
BATCH_SIZE = 256  # 批处理大小
CELL_SIZE = 256  # 用于GRU的单元大小
DROPOUT_RATE = 0.85  # 用于CNN的dropout率
LEARNING_RATE = 1e-3  # 学习率
NODE_SIZE = [512, 256, 128]  # MLP中每层的节点数
NUM_LAYERS = 5  # MLP中的层数

# 解析命令行参数的函数
def parse_args():
    # 创建解析器
    parser = argparse.ArgumentParser(
        description='''
██╗      █████╗ ███████╗ █████╗ ██████╗ ██╗   ██╗███████╗
██║     ██╔══██╗╚══███╔╝██╔══██╗██╔══██╗██║   ██║██╔════╝
██║     ███████║  ███╔╝ ███████║██████╔╝██║   ██║███████╗
██║     ██╔══██║ ███╔╝  ██╔══██║██╔══██╗██║   ██║╚════██║
███████╗██║  ██║███████╗██║  ██║██║  ██║╚██████╔╝███████║
╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝        
使用深度学习方法研究恶意代码家族识别
'''
    )
    # 添加参数组
    group = parser.add_argument_group("Arguments")
    # 添加模型选择参数
    group.add_argument("-m", "--model", required=True, type=int, help="选择模型：1 CNN-SVM, 2 GRU-SVM, 3 MLP-SVM")
    # 添加数据集参数
    group.add_argument("-d", "--dataset", required=True, type=str, help="指定数据集路径")
    # 添加训练轮数参数
    group.add_argument("-n", "--num_epochs", required=True, type=int, help="指定训练轮数")
    # 添加惩罚参数
    group.add_argument("-c", "--penalty_parameter", required=True, type=float, help="指定惩罚参数")
    # 添加训练模型保存路径参数
    group.add_argument("-k", "--checkpoint_path", required=True, type=str, help="指定训练模型保存路径")
    # 添加Tensorflow日志保存路径参数
    group.add_argument("-l", "--log_path", required=True, type=str, help="指定Tensorflow日志保存路径")
    # 添加训练结果保存路径参数
    group.add_argument("-r", "--result_path", required=True, type=str, help="指定训练结果保存路径")
    # 解析命令行参数
    arguments = parser.parse_args()
    return arguments

# 主函数，用于执行模型训练
def main(arguments):
    # 获取模型选择
    model_choice = arguments.model
    # 确保模型选择是有效的
    assert model_choice in [1, 2, 3], "Invalid choice: Choose among 1, 2, and 3 only."

    # 加载数据集
    dataset = np.load(arguments.dataset)

    # 预处理数据集
    features, labels = load_data(dataset=dataset)
    labels = one_hot_encode(labels=labels)

    # 获取特征数量和类别数量
    num_features = features.shape[1]
    num_classes = labels.shape[1]

    # 将数据集分割为训练集和测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.30, stratify=labels
    )

    # 调整训练集和测试集的大小以适应批处理
    train_size = int(train_features.shape[0])
    train_features = train_features[: train_size - (train_size % BATCH_SIZE)]
    train_labels = train_labels[: train_size - (train_size % BATCH_SIZE)]

    test_size = int(test_features.shape[0])
    test_features = test_features[: test_size - (test_size % BATCH_SIZE)]
    test_labels = test_labels[: test_size - (test_size % BATCH_SIZE)]

    # 根据选择的模型进行训练
    if model_choice == 1:
        # 创建CNN模型实例
        model = CNN(
            alpha=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            num_classes=num_classes,
            penalty_parameter=arguments.penalty_parameter,
            sequence_length=num_features,
        )
        # 训练CNN模型
        model.train(
            checkpoint_path=arguments.checkpoint_path,
            log_path=arguments.log_path,
            result_path=arguments.result_path,
            epochs=arguments.num_epochs,
            train_data=[train_features, train_labels],
            train_size=int(train_features.shape[0]),
            test_data=[test_features, test_labels],
            test_size=int(test_features.shape[0]),
        )
    elif model_choice == 2:
        # 调整数据形状以适应GRU模型
        train_features = np.reshape(train_features, (train_features.shape[0], int(np.sqrt(train_features.shape[1])), int(np.sqrt(train_features.shape[1]))))
        test_features = np.reshape(test_features, (test_features.shape[0], int(np.sqrt(test_features.shape[1])), int(np.sqrt(test_features.shape[1]))))

        # 创建GRU模型实例
        model = GruSvm(
            alpha=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            cell_size=CELL_SIZE,
            dropout_rate=DROPOUT_RATE,
            num_classes=num_classes,
            num_layers=NUM_LAYERS,
            sequence_height=train_features.shape[2],
            sequence_width=train_features.shape[1],
            svm_c=arguments.penalty_parameter,
        )
        # 训练GRU模型
        model.train(
            checkpoint_path=arguments.checkpoint_path,
            log_path=arguments.log_path,
            epochs=arguments.num_epochs,
            train_data=[train_features, train_labels],
            train_size=int(train_features.shape[0]),
            test_data=[test_features, test_labels],
            test_size=int(test_features.shape[0]),
            result_path=arguments.result_path,
        )
    elif model_choice == 3:
        # 创建MLP模型实例
        model = MLP(
            alpha=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            node_size=NODE_SIZE,
            num_classes=num_classes,
            num_features=num_features,
            penalty_parameter=arguments.penalty_parameter,
        )
        # 训练MLP模型
        model.train(
            checkpoint_path=arguments.checkpoint_path,
            num_epochs=arguments.num_epochs,
            log_path=arguments.log_path,
            train_data=[train_features, train_labels],
            train_size=int(train_features.shape[0]),
            test_data=[test_features, test_labels],
            test_size=int(test_features.shape[0]),
            result_path=arguments.result_path,
        )

# 程序入口点
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    # 执行主函数
    main(args)