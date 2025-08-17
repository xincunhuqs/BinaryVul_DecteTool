# 一、工具简介：

​        BinaryVul_DecteTool是一款基于深度学习结合内联汇编技术，实现的二进制漏洞检测工具，目前针对5k+个缺陷汇编代码块，使用Transformer神经网络深度学习模型，进行缺陷特征学习，支持检测71种二进制文件漏洞缺陷。该工具初代版本是一个CLI工具，使用方式简便，目前支持的参数为：efp,efdp,nrsc,acsc四个参数，具体使用教程见：工具使用教程。

<font color=red>注意：目前该工具仅支持对PE格式的二进制文件进行缺陷检测。</font>

工具获取：
 由于改项目中存在一个大于180MB的文件，因此下载该项目时，只能通过git clone方式下载，执行如下命令：
```
git clone  git clone https://github.com/xincunhuqs/BinaryVul_DecteTool.git
```
<img width="1066" height="592" alt="image" src="https://github.com/user-attachments/assets/d4e02c71-cda7-49a7-9c6d-304c75c86166" />


# 二、工具使用教程：

进入Bvsc_tool目录，执行BVSC.py代码，命令如下:

```
python BVSC.py --help
```

<img width="1250" height="744" alt="image" src="https://github.com/user-attachments/assets/1671e81f-3f38-4938-9eb0-e7824dc9d239" />


参数解释：

 -efp：全称为exefile_path，表示待检测的二进制文件的路径；

-efdp：全称为exefile_folder_path，当需要检测一个二进制文件夹时，可以使用该参数传入待检测二进制文件夹的路径；

-nrsc：全称为normal_scan，该参数表示对二进制文件进行默认扫描，调用本地训练好的深度学习模型进行二进制文件缺陷识别，检测结果保存在同目录下的singchecked_result文件中。该参数的默认值为True；

-acsc: 全称为：accurate_scan，该参数表示调用Deepseek模型对预测的结果进行研判，并且给出分析结果。检测结果保存在同目录下的singchecked_result文件中，该参数默认值为False；

-v:全称为：verbose，BinaryVul_DecteTool在扫描过程中默认以静默模式运行，不输出过程信息。将参数-v设置为True，可展示扫描过程中的相关信息。

# 三、BinaryVul_DecteTool工具使用实例

### 1、扫描单个二进制文件

​    可以使用 -efp 指定需要扫描的二进制文件，python BVSC.py -efp "check_binaryfile_path"

```
python BVSC.py -efp "check_binaryfile_path" [-v] [-acsc]
 ```


​    <img width="1950" height="551" alt="image" src="https://github.com/user-attachments/assets/1e312206-ec64-4851-b366-2624e41ebff0" />



**使用-acsc参数调用大模型进行解析结果降噪，使用-v参数展示扫描过程信息。**
<img width="1934" height="574" alt="image" src="https://github.com/user-attachments/assets/e60a7411-bf28-4bbd-ba24-02623b1b8eea" />


### 2、扫描二进制文件夹

可使用-efdp参数指定需要扫描的二进制文件夹，python BVSC.py -efdp "check_binaryfile_folder"

```
python BVSC.py -efdp "check_binaryfile_folder" [-v] [-acsc]
```

<img width="1936" height="434" alt="image" src="https://github.com/user-attachments/assets/8fdc0ca0-ca98-4cc4-9639-94da8aa145ed" />

<img width="1967" height="280" alt="image" src="https://github.com/user-attachments/assets/3eb052d8-8257-4f72-a81a-4d3f8fc7f8c2" />



**使用-acsc参数调用大模型进行解析结果降噪，使用-v参数展示扫描过程信息。**

<img width="1943" height="347" alt="image" src="https://github.com/user-attachments/assets/71c480a3-69ac-4e3a-93f0-060bfe27a239" />

<img width="1964" height="476" alt="image" src="https://github.com/user-attachments/assets/a9c566e3-1b27-4c0d-9759-e59a2e3a2582" />
