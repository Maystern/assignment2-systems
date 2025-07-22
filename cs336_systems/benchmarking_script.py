import argparse
import torch
import timeit
import pandas as pd

from cs336_basics.transformer import Transformer

device = "cpu"

if __name__ == "__main__":
    vocab_size = 10_000
    batch_size = 4
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--context_length', '-l', help='模型捕获的上下文长度')
    parser.add_argument('--d_model', '-d', help='模型的隐空间大小')
    parser.add_argument('--d_ff', '-f', help='模型的全连接层的提升的隐空间大小')
    parser.add_argument('--num_layers', '-t', help='模型 transformer 块的数量')
    parser.add_argument('--num_heads', '-s', help='模型 attention 块中注意力头的数量')

    args = parser.parse_args()

    model = Transformer(
        vocab_size=int(vocab_size),
        context_length=int(args.context_length),
        d_model=int(args.d_model),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        d_ff=int(args.d_ff),
        rope_theta=10000,
        device=torch.device(device)
    )
    
    # uv run python cs336_systems/benchmarking_script.py -l 1024 -d 768 -f 3072 -t 12 -s 12
    total_params_cnt = 0
    total_mem_cost = 0
    for param in model.parameters():
        total_params_cnt += param.numel()
        total_mem_cost += param.numel() * param.element_size()
    
    print(f"模型参数: {total_params_cnt / 1000 / 1000:.2f} M")
    print(f"模型内存: {total_mem_cost / 1024 / 1024:.2f} MB")

    X = torch.randint(0, vocab_size, (batch_size, int(args.context_length))).to(torch.device(device))
    print(X)
    print(X.shape)

    warm_up_steps = 5
    measure_steps = 10
    warm_up_time_list = []
    measure_time_list = []
    for epoch in range(warm_up_steps + measure_steps):
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = timeit.default_timer()
        out = model(X)
        if device == "cuda":
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        if epoch < warm_up_steps:
            warm_up_time_list.append(end_time - start_time)
            print(f"[warm up] 第 {epoch + 1} 次计算耗时, 耗时 {end_time - start_time:.2f} s")
        else:
            measure_time_list.append(end_time - start_time)
            print(f"[measure] 第 {epoch - warm_up_steps + 1} 次计算耗时, 耗时 {end_time - start_time:.2f} s")
    
    df = pd.DataFrame({
        "warm up": warm_up_time_list,
        "measure": measure_time_list
    })
    print(df)