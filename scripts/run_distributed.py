#!/usr/bin/env python3
"""
多机多卡训练启动脚本
支持SLURM和手动启动两种方式
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def setup_slurm_environment():
    """设置SLURM环境变量"""
    # 获取SLURM环境变量
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    node_rank = int(os.environ["SLURM_NODEID"])
    
    # 设置分布式训练环境变量
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["NODE_RANK"] = str(node_rank)
    
    # 设置主节点地址和端口
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "localhost")
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    
    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "node_rank": node_rank
    }


def setup_manual_environment(args):
    """设置手动启动的环境变量"""
    os.environ["RANK"] = str(args.rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["LOCAL_RANK"] = str(args.local_rank)
    os.environ["NODE_RANK"] = str(args.node_rank)
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    
    return {
        "rank": args.rank,
        "world_size": args.world_size,
        "local_rank": args.local_rank,
        "node_rank": args.node_rank
    }


def create_slurm_script(args):
    """创建SLURM提交脚本"""
    script_content = f"""#!/bin/bash
    #SBATCH --job-name={args.job_name}
    #SBATCH --nodes={args.num_nodes}
    #SBATCH --ntasks-per-node={args.gpus_per_node}
    #SBATCH --cpus-per-task={args.cpus_per_task}
    #SBATCH --gres=gpu:{args.gpus_per_node}
    #SBATCH --time={args.time_limit}
    #SBATCH --output={args.output_dir}/slurm_%j.out
    #SBATCH --error={args.output_dir}/slurm_%j.err
    #SBATCH --partition={args.partition}

    # 设置环境变量
    export MASTER_PORT=29500
    export MASTER_ADDR=$SLURM_LAUNCH_NODE_IPADDR

    # 激活conda环境（如果需要）
    # source activate your_env_name

    # 运行训练脚本
    srun python {args.script_path} {args.script_args}
    """
    
    script_path = Path(args.output_dir) / "run_slurm.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod(script_path, 0o755)
    
    print(f"SLURM脚本已创建: {script_path}")
    print(f"使用以下命令提交作业:")
    print(f"sbatch {script_path}")


def run_training_directly(script_args):
    """直接运行训练，而不是通过subprocess"""
    print(f"直接运行训练，参数: {script_args}")
    
    # 导入训练模块
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # 导入训练相关模块
    from scripts import train
    import openpi.training.config as _config
    
    # 设置命令行参数
    sys.argv = ["train.py"] + script_args
    
    # 获取配置并运行
    config = _config.cli()
    train.main(config)


def main():
    parser = argparse.ArgumentParser(description="多机多卡训练启动脚本")
    parser.add_argument("--mode", choices=["slurm", "manual", "create_slurm"], 
                       default="manual", help="运行模式")
    parser.add_argument("--direct", action="store_true", 
                       help="直接运行训练，不使用subprocess")
    
    # SLURM脚本创建参数
    parser.add_argument("--job-name", default="distributed_training", help="作业名称")
    parser.add_argument("--num-nodes", type=int, default=2, help="节点数量")
    parser.add_argument("--gpus-per-node", type=int, default=4, help="每个节点的GPU数量")
    parser.add_argument("--cpus-per-task", type=int, default=8, help="每个任务的CPU数量")
    parser.add_argument("--time-limit", default="24:00:00", help="时间限制")
    parser.add_argument("--partition", default="gpu", help="分区名称")
    parser.add_argument("--output-dir", default="./slurm_scripts", help="输出目录")
    parser.add_argument("--script-path", default="scripts/train.py", help="训练脚本路径")
    parser.add_argument("--script-args", default="", help="训练脚本参数")
    
    # 手动启动参数
    parser.add_argument("--rank", type=int, default=0, help="当前进程的rank")
    parser.add_argument("--world-size", type=int, default=4, help="总进程数")
    parser.add_argument("--local-rank", type=int, default=0, help="本地rank")
    parser.add_argument("--node-rank", type=int, default=0, help="节点rank")
    parser.add_argument("--master-addr", default="10.10.1.16", help="主节点地址")
    parser.add_argument("--master-port", type=int, default=29500, help="主节点端口")
    
    args = parser.parse_args()
    
    if args.mode == "create_slurm":
        create_slurm_script(args)
        return
    
    # 设置环境变量
    if args.mode == "slurm":
        dist_info = setup_slurm_environment()
    else:
        dist_info = setup_manual_environment(args)
    
    print(f"分布式训练环境设置完成:")
    print(f"  Rank: {dist_info['rank']}/{dist_info['world_size']}")
    print(f"  Local Rank: {dist_info['local_rank']}")
    print(f"  Node Rank: {dist_info['node_rank']}")
    print(f"  Master Addr: {os.environ.get('MASTER_ADDR', 'localhost')}")
    print(f"  Master Port: {os.environ.get('MASTER_PORT', '29500')}")
    
    # 设置CUDA设备
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(dist_info["local_rank"])
    
    # 设置JAX环境变量
    os.environ["JAX_PLATFORM_NAME"] = "gpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    
    # 运行训练脚本
    if len(sys.argv) > 1 and sys.argv[1] == "--":
        # 如果使用 -- 分隔符，后面的参数都是训练脚本的参数
        script_args = sys.argv[2:]
    else:
        # 否则使用默认参数
        script_args = [
            "pi0_bridge_traj",
            "--exp-name", "pi0_bridge_traj",
            "--overwrite",
        ]
    
    if args.direct:
        # 直接运行训练
        run_training_directly(script_args)
    else:
        # 通过subprocess运行
        cmd = [sys.executable, "scripts/train.py"] + script_args
        
        print(f"执行命令: {' '.join(cmd)}")
        
        # 运行训练脚本
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"训练脚本执行失败: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("训练被用户中断")
            sys.exit(0)


if __name__ == "__main__":
    main()