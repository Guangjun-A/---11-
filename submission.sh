# 离线去除无效数据
python data/remove_zero_frame.py

# 生成提交文件
python submission.py --eval_model_path final_test_B --seg 500 -k 10 -c None