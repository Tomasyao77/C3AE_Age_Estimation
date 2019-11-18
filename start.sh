#! /bin/bash
#项目根目录
basepath="/media/gisdom/data11/tomasyao/workspace/pycharm_ws/C3AE_Age_Estimation"
DATE=`date +%Y%m%d_%H%M%S`

if [ $# -ne 1 ] #有且仅有一个参数，否则退出
then
	echo "Usage: /start.sh train[|val]"
	exit 1
else
	echo "starting..."
fi

if [ $1 = "train" ]
then
	#后台运行训练代码
	echo "train..."
	source activate tfgpu
	cd ${basepath}/examples
	setsid python ./multi_gpus_train.py > ${basepath}/logs/${DATE}_train_log_50_pre 2>&1 &
elif [ $1 = "val" ]
then
	echo "validate..."
	source activate tfgpu
	cd ${basepath}/examples
	python ./multi_gpus_val.py
# elif [ $1 = "demo" ]
# then
# 	echo "demo..."
# 	source activate torch #cpu only
# 	python ${basepath}/demo.py --img_dir=${basepath}/img_dir --output_dir=${basepath}/output_dir --resume=${basepath}/misc/epoch044_0.02343_3.9984.pth
# elif [ $1 = "tboard" ]
# then
# 	tensorboard --logdir=${basepath}/tf_log
else
	echo "do nothing"
fi