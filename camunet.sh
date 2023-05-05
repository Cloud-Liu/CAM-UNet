
export CUDA_VISIBLE_DEVICES=1;
nohup     python train.py --dataset ISIC2018   --arch CAM_UNet --name archs_camunet_23_2_21_10_30  --img_ext .jpg --mask_ext .png --lr 0.01 --epochs 200 --input_w 512 --input_h 512 --b 4   --num_workers 16 --deep_supervision True > logs/train_archs_camunet_23_2_21_10_30.log 2>&1 &

