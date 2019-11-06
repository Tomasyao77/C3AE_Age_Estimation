#! /bin/bash
#项目根目录
basepath="/media/gisdom/data11/tomasyao/workspace/pycharm_ws/age-estimation-pytorch"

if [ $# -ne 1 ] #有且仅有一个参数，否则退出
then
	echo "Usage: /start.sh train[|test|demo]"
	exit 1
else
	echo "starting..."
fi

if [ $1 = "train" ]
then
	#后台运行训练代码
	echo "train..."
	source activate torchg
	setsid python ${basepath}/train.py --data_dir=${basepath}/data_dir/appa-real-release --tensorboard=${basepath}/tf_log --checkpoint=${basepath}/checkpoint > /tmp/log_t 2>&1 &
elif [ $1 = "test" ]
then
	echo "test..."
	source activate torchg
	#测试速度很快所以就不在后台运行了
	python ${basepath}/test.py --data_dir=${basepath}/data_dir/FG-NET --resume=${basepath}/checkpoint/epoch050_0.02353_4.0066.pth
elif [ $1 = "demo" ]
then
	echo "demo..."
	source activate torch #cpu only
	python ${basepath}/demo.py --img_dir=${basepath}/img_dir --output_dir=${basepath}/output_dir --resume=${basepath}/misc/epoch044_0.02343_3.9984.pth
elif [ $1 = "tboard" ]
then
	tensorboard --logdir=${basepath}/tf_log
else
	echo "do nothing"
fi