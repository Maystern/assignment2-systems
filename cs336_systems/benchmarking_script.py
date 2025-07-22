import argparse

from cs336_basics.transformer import Transformer
context_length = 1024 # 变量
d_model = 768 # 变量
d_ff = 3072 # 变量
num_layers = 12 # 变量
num_heads = 12 # 变量

if __name__ == "__main__":
    vocab = 10_000
    batch_size = 4
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--context_length', '-l', help='模型捕获的上下文长度')
    parser.add_argument('--d_model', '-dm', help='模型的隐空间大小')
    parser.add_argument('--d_ff', '-dff', help='模型的全连接层的提升的隐空间大小')
    parser.add_argument('--num_layers', '-t', help='模型 transformer 块的数量')
    parser.add_argument('--num_heads', '-h', help='模型 attention 块中注意力头的数量')

    args = parser.parse_args()

    model = cs336_basics.transformer.Transformer(

    )