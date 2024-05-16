
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from utils import data
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

MALWARE_FAMILIES = [
    "Adialer.C",
    "Agent.FYI",
    "Allaple.A",
    "Allaple.L",
    "Alueron.gen!J",
    "Autorun.K",
    "C2LOP.P",
    "C2LOP.gen!g",
    "Dialplatform.B",
    "Dontovo.A",
    "Fakerean",
    "Instantaccess",
    "Lolyda.AA1",
    "Lolyda.AA2",
    "Lolyda.AA3",
    "Lolyda.AT",
    "Malex.gen!J",
    "Obfuscator.AD",
    "Rbot!gen",
    "Skintrim.N",
    "Swizzor.gen!E",
    "Swizzor.gen!I",
    "VB.AT",
    "Wintrim.BX",
    "Yuner.A",
]


def parse_args():
    parser = argparse.ArgumentParser(
description=''' 
██╗      █████╗ ███████╗ █████╗ ██████╗ ██╗   ██╗███████╗
██║     ██╔══██╗╚══███╔╝██╔══██╗██╔══██╗██║   ██║██╔════╝
██║     ███████║  ███╔╝ ███████║██████╔╝██║   ██║███████╗
██║     ██╔══██║ ███╔╝  ██╔══██║██╔══██╗██║   ██║╚════██║
███████╗██║  ██║███████╗██║  ██║██║  ██║╚██████╔╝███████║
╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
使用深度学习方法研究恶意代码家族识别                                                      
''', formatter_class=argparse.RawTextHelpFormatter
    )
    group = parser.add_argument_group("Arguments")
    group.add_argument(
        "-r",
        "--result_path",
        required=True,
        type=str,
        help="指定存放结果的路径",
    )
    group.add_argument(
        "-t",
        "--figure_title",
        required=True,
        type=str,
        help="指定混淆矩阵的标题",
    )
    arguments = parser.parse_args()
    return arguments


def main(arguments):
    conf, acc, report = data.plot_confusion_matrix(
        arguments.figure_title, arguments.result_path, MALWARE_FAMILIES
    )

    print("{} 识别结果报告:\n{}".format(arguments.figure_title, report))

    print("{} 混淆矩阵:\n{}".format(arguments.figure_title, conf))

    print("{} 精确度: {}".format(arguments.figure_title, acc))


if __name__ == "__main__":
    args = parse_args()

    main(arguments=args)
