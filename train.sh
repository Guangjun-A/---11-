

# 打乱数据并删除无效帧（一帧内全为0或者全为NULL的数据为无效帧）。
# 生成
python data/remove_zero_frame.py


# 对六个模型进行10折交叉验证，然后从每一折中选出准确率最高的模型（共有10个）进行stack boosting.
for idx in  0 1 2 3 4 5 6 7 8 9 # 1 2 3 4 5 6 7 8 9
do
  echo $idx
  python train.py --config configs/ensemble/${idx}.yaml  --seg 500  -b 4 --focal_loss -k 10
done
