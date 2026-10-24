'''
-basics
-test
-benchmark

'''
import torch
import triton
import triton.language as tl


DEVICE =  torch.device(f'cuda:{torch.cuda.current_device()}')


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE:tl.constexpr,    #告诉 Triton，这个参数是编译期常量
):
    PID = tl.program_id(axis=0) #获取的是 Program 的编号,当前 Triton Program 在 Grid 中的索引（编号）,当前属于哪一个block

    block_start = PID * BLOCK_SIZE
    offsets = block_start + tl.arange(0,BLOCK_SIZE)
    mask = offsets < n_elements

    #加载数据from DRAM/HBM/VRAM  to  SRAM

    #other=None：对于未读取的位置，返回值是未定义的（undefined），不能保证是 0 或其他特定数值。
    x = tl.load(x_ptr + offsets,mask=mask,other=None)
    y = tl.load(y_ptr + offsets,mask=mask,other=None)


    output = x+y


    #将数据写回DrAM
    tl.store(output_ptr+offsets,output,mask=mask)

def add(x,y):

    output = torch.empty_like(x)

    assert x.device == DEVICE and y.device == DEVICE

    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements,meta['BLOCK_SIZE']),)   #(4,)
    #其中“cdiv”是向上取整的除法，grid指的是网格大小，即：有多少个block
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=1024
    )
    return output





def test_add_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    # create test data
    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE)
    y = torch.randn(size, device=DEVICE)

    # run triton kernel & pytorch equivalent
    z_tri = add(x, y)   #triton写的加法算子
    z_ref = x + y

    # compare
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("passed")

#do_bench() 会重复执行多次完整的向量加法，每次都计算 n_elements 个元素，然后根据每次的耗时统计 P50、P90、P99
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 21)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="vector-add",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.randn(size, device="cuda")
    y = torch.randn(size, device="cuda")
    quantiles = [0.9, 0.5, 0.99]

    if provider == "triton":
        min_ms,ms,max_ms = triton.testing.do_bench(lambda: add(x, y),quantiles=quantiles)
    else:
        min_ms,ms,max_ms = triton.testing.do_bench(lambda: x + y,quantiles=quantiles)
    gbps = lambda ms: 3*x.numel()*x.element_size()*1e-9 /(ms*1e-3)
    return gbps(min_ms),gbps(ms),gbps(max_ms)


#benchmark.run(print_data=True, show_plots=True)




if __name__ == "__main__":
    test_add_kernel(size=4096)
    test_add_kernel(size=4097)
    test_add_kernel(size=98432)
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path=".", print_data=True, show_plots=True)
