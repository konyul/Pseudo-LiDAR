CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --cfg_file ./cfgs/kitti_models/CaDDN_ATSS_W.yaml
#CUDA_VISIBLE_DEVICES=2 python3 train.py --cfg_file ./cfgs/kitti_models/CaDDN_PL_FS.yaml

#CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --cfg_file ./cfgs/kitti_models/CaDDN.yaml
#CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch --cfg_file ./cfgs/kitti_models/CaDDN_W.yaml --ckpt ../output/cfgs/kitti_models/CaDDN_W/default/ckpt --eval_all
python3 alarm.py
#"--ckpt","baseline/cfgs/kitti_models/CaDDN/default/ckpt/checkpoint_epoch_80.pth"
