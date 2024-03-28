PYTHONPATH=$PYTHONPATH:../../ \
# srun --mpi=pmi2 -p$2 -n$1 --gres=gpu:$1 --ntasks-per-node=$1 --cpus-per-task=4 --job-name=mvtec \
export MASTER_ADDR="127.0.0.1"

# 通信に使用するポート番号（適宜変更してください）
export MASTER_PORT="12345"

# このプロセスのランク（単一GPUのため0を設定）
export RANK=0
export LOCAL_RANK=0

# 総ワーカー数（GPUが1つのみなので1を設定）
export WORLD_SIZE=1
/home/sakai/projects/Reimpl/HVQ-Trans/hvq-trans/bin/python3 -u ./tools/train_val.py --config /home/sakai/projects/Reimpl/HVQ-Trans/HVQ-Trans/experiments/MVTec-AD/config.yaml
