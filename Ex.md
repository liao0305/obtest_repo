### 实验过程记录

### 1、登陆AutoDL

https://www.autodl.com/console/instance/list平台，购买容器实例

![image-20240518181626970](C:\Users\Coke\AppData\Roaming\Typora\typora-user-images\image-20240518181626970.png)



### 2、SSH远程登录

````
ssh -p 52589 root@connect.westb.seetacloud.com  
密码：Fx5bVv6ONaKk
````

![image-20240518183356774](C:\Users\Coke\AppData\Roaming\Typora\typora-user-images\image-20240518183356774.png)



### 3、环境首次准备（若已配置好，则跳过）  

```shell
conda create --name openmmlab python=3.8 -y 
conda activate openmmlab 

conda install pytorch torchvision -c pytorch 
pip install -U openmim 
mim install mmengine 
mim install mmcv 
mim install mmdet 

# 源码编译 
git clone https://github.com/open-mmlab/mmocr.git cd mmocr pip install -v -e . 

# 设置清华镜像源 
pip config set global.index-url  https://pypi.tuna.tsinghua.edu.cn/simple
```



### 4、vscode配置远程开发

* 安装插件Remote - SSH 
* 添加配置

```shell
Host AutoDL
 HostName connect.westb.seetacloud.com
 User root
 IdentityFile C:\Users\LtCc\.ssh\id_rsa
```

* 设置免密登陆

```
# 生成公钥
ssh-keygen -t rsa -b 4096
 # 复制到远程服务器上
cd .ssh/
 vim authorized_keys
```



### 5、训练模型1（以ICDAR 2015为例）

#### 5.1、ICDAR 2015 数据集

```
conda info --envs
conda activate openmmlab


python tools/dataset_converters/prepare_dataset.py icdar2015 --task textdet


ll data/icdar2015/
```

![image-20240518183750650](C:\Users\Coke\AppData\Roaming\Typora\typora-user-images\image-20240518183750650.png)



* 可视化数据集的标签是否被正确生成

```
python tools/visualizations/browse_dataset.py configs/textdet/_base_/datasets/icdar2015.py
```

#### 5.2、ICDAR 2015 训练

```
python tools/train.py configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py
```

![image-20240518184006422](C:\Users\Coke\AppData\Roaming\Typora\typora-user-images\image-20240518184006422.png)

#### 5.3、预测测试集

```
python tools/test.py configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py work_dirs/dbnet_resnet18_fpnc_1200e_icdar2015/epoch_800.pth
```

![image-20240518184530348](C:\Users\Coke\AppData\Roaming\Typora\typora-user-images\image-20240518184530348.png)



![image-20240518184551422](C:\Users\Coke\AppData\Roaming\Typora\typora-user-images\image-20240518184551422.png)



### 6、训练模型2（以cute80数据集为例）

#### 6.1、数据集准备(若数据集已下载，则跳过此步骤)

```
python tools/dataset_converters/prepare_dataset.py cute80 --task textrecog
```

#### 6.2、训练

```
python tools/train.py configs/textrecog/abinet/abinet_20e_st-an_mj.py
```

#### 6.3、测试

```
python tools/infer.py data/cute80/textrecog_imgs/test/023.jpg --rec ABINet --print-result --save_pred --save_vis

python tools/infer.py data/cute80/textrecog_imgs/test/024.jpg --rec ABINet --print-result --save_pred --save_vis

python tools/infer.py data/cute80/textrecog_imgs/test/042.jpg --rec ABINet --print-result --save_pred --save_vis
```



### X、常见问题记录

#### 1、处理磁盘空间不足的问题

```
 mv /root/mmocr/icdar2015/ /root/autodl-tmp/
 mv /root/mmocr/imgs/vis_data/ /root/autodl-tmp/


```

#### 2、Python代码报错

```
Obtaining train Dataset...
 ctw1500_train_images has been extracted. Skip
 ctw1500_train_labels has been extracted. Skip
 Gathering train Dataset...
 Parsing train Images and Annotations...
 [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 
1000/1000, 4354.0 task/s, elapsed: 0s, ETA:     
Packing train Annotations...
 [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 
1000/1000, 68.4 task/s, elapsed: 15s, ETA:     
Dumping train Annotations...
 Obtaining test Dataset...
 ctw1500_test_images has been extracted. Skip
 Gathering test Dataset...
 Parsing test Images and Annotations...
 [                                                  
0s
 0s
 ] 
0/500, elapsed: 0s, 
ETA:multiprocessing.pool.RemoteTraceback:
 """
 Traceback (most recent call last):
 File 
"/root/miniconda3/envs/openmmlab/lib/python3.8/multipro
 cessing/pool.py", line 125, in worker
 result = (True, func(*args, **kwds))
 File 
"/root/miniconda3/envs/openmmlab/lib/python3.8/multipro
 cessing/pool.py", line 51, in starmapstar
 return list(itertools.starmap(args[0], args[1]))
 File 
"/root/mmocr/mmocr/datasets/preparers/parsers/ctw1500_p
 arser.py", line 57, in parse_file
 instances = self.load_txt_info(ann_path)
 File 
"/root/mmocr/mmocr/datasets/preparers/parsers/ctw1500_p
 arser.py", line 69, in load_txt_info
 for line in list_from_file(anno_dir):

```

解决方法：

```
import os
 def add_prefix(directory, prefix):
 for filename in os.listdir(directory):
 if not os.path.isfile(os.path.join(directory, 
filename)):
 continue
 new_filename = f"{prefix}{filename}"
 old_path = os.path.join(directory, filename)
 new_path = os.path.join(directory, 
new_filename)
 try:
 os.rename(old_path, new_path)
 print(f"已为 {new_filename} 添加了前缀")
 except FileExistsError:
 print(f"无法重命名 {filename}: 文件已存在")
 # 调用函数并传入要处理的目录路径和前缀字符串
add_prefix("./", "000")
```



