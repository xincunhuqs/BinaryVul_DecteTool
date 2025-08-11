import os
import ast
import re
from shutil import rmtree

"""
# @Time    : 2025/07/17
# @Author  : huqinsong
# @Version: 自定义分词器
# @Desc: 主要实现：
        自定义分词器，实现自动化创建词典
"""


def vocabularygenerate(defactfile_path,max_vocabsize=1000):
    """
    对传入的文件进行分组处理，构成初始的tokenizer字典
    defactfile_path: 缺陷代码文件
    """

    vocabulary_dict={}
    vocabulary_dict["pad"]=0
    index=len(vocabulary_dict)+2
    with open(defactfile_path, 'r',encoding="utf-8") as fdata:
        file_content_linelsit=fdata.read().split("\n")
    number_flage = True
    for line in file_content_linelsit:
        for word in line.split(" "):
            if "0x" in word :
                if number_flage:
                    vocabulary_dict["0x_word"]=1
                    number_flage = False
                continue
            if word not in vocabulary_dict:
                index+=1
                vocabulary_dict[word]=index
    padding_vocabulary = {"unknow_key"+str(index):2 for index in range(1,max_vocabsize-len(vocabulary_dict))}
    vocabulary_dict.update(padding_vocabulary)
    if os.path.exists(r"./tokenize_dict.txt"):
        os.remove(r"./tokenize_dict.txt")
    with open("./tokenize_dict.txt","a",encoding="utf-8") as fdata_dice:
        fdata_dice.write(str(vocabulary_dict)+"\n")
        # print("jilu", vocabulary_dict)
    # print(len(vocabulary_dict.keys()))
    # print("asdf",vocabulary_dict)
    # print("分词结束！！")
    return vocabulary_dict


def word2index(sentcence,tokenizer_dist,max_sentencelength=500):
    #将句子转换为对应的索引
    """
    对传入的句子进行序列化
    sentcence:待序列化的汇编语句代码
    tokenizer_dist:序列化使用到的字典
    max_sentencelength:待转换汇编语句的最大长度

    """
    if max_sentencelength:
        stenceinitialize_list=["pad" for _ in range(max_sentencelength-len(sentcence.split(" ")))]
        sentcence_lsit=sentcence.split(" ")+stenceinitialize_list
    else:
        sentcence_lsit=sentcence.split(" ")
    temp_index=0
    updata_flag=False
    pattern = re.compile("unknow_key")
    global unknow_index
    for word in sentcence_lsit:
        if word in tokenizer_dist.keys():
            sentcence_lsit[temp_index]=tokenizer_dist[word]
        else:
            if "0x" in word:
                word = "0x_word"
                sentcence_lsit[temp_index] = tokenizer_dist[word]
            else:  
                #直接替换字典中key的值
                # print("当前未知词为：",word)
                updata_flag = True
                tokenizer_key_list = list(tokenizer_dist.keys())
                tokenizer_value_list = list(tokenizer_dist.values())
                # print(tokenizer_value_list)
                for index,key in enumerate(tokenizer_key_list):
                    if pattern.search(key):
                        unknow_index = index
                        break
                tokenizer_key_list[unknow_index]=word
                tokenizer_dist={key:tokenizer_key_list.index(key) for key in tokenizer_key_list}
                sentcence_lsit[temp_index]=(tokenizer_dist[word])
        temp_index+=1
    if updata_flag :
        # print("词表已经更新")
        updatatokenizer(tokenizer_dist)
        # print(loadtokenizer("./tokenize_dict.txt"))
    return sentcence_lsit

def loadtokenizer(token_path):
    """
    实现从指定文件中读取数据，构成字典。
    token_path:字典文件所在文件路径
    """
    with open(token_path, 'r', encoding="utf-8") as fdata:
       tokenizer_dist = fdata.readlines()[-1].strip("\n")
    # print("读取到的字典内容为：",tokenizer_dist)
    return ast.literal_eval(tokenizer_dist)


def updatatokenizer(tokenizer_dist):
    """
    对传入的tokenizer_dist进行处理，并且更新到最新字典记录文件中
    tokenizer_dist:更新的字典内容
    """

    with open("./tokenize_dict.txt","w",encoding="utf-8") as ftoken:
        ftoken.write(str(tokenizer_dist)+"\n")
    # print("字典数据更新结束！！")


if __name__ == '__main__':
    vocabularygenerate(r"./data/train.txt")
    tokenizer_dist=loadtokenizer(r".\tokenize_dict.txt")
    print(tokenizer_dist)

    # print("词表大小为：",len(list(tokenizer_dist)))
    stence="cmp dword ptr [ 0x423088 ], 0; je 0x4018f9;mov dword ptr [ebp - 4], 0;cmp dword ptr [0x423088], 0; je 0x40191f E  E"
    result=word2index(stence,tokenizer_dist,50)
    print(result)
    key_list=[index for index in stence.split(" ")]
    print(key_list)
    # for key in key_list:
    #     print(tokenizer_dist[key],key)
    # print(len(list(loadtokenizer(r"D:\graduation_project\loacal_tokennize\tokenize_dict1.txt"))))
    # loadtokenizer("./tokenize_dict1.txt")
