#实现本地的分词器
import os
import ast
import re


def vocabularygenerate(defactfile_path,max_vocabsize=1000):
    """
    使用传入的文件进行分词处理，构成初始的tokenizer字典
    defactfile_path:缺陷代码文件
    """
    vocabulary_dict={}
    vocabulary_dict["pad"]=0
    index=len(vocabulary_dict)+2
    with open(defactfile_path, 'r',encoding="utf-8") as fdata:
        file_content_linelsit=fdata.read().split("\n")
    number_flage = True
    for line in file_content_linelsit:
        new_line=line.replace(","," ").replace(":"," ").replace(";","; ")\
            .replace("\t\t"," ").replace("["," [ ").replace("]"," ] ")
        for word in new_line.split(" "):
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
    with open("./tokenize_dict.txt","w",encoding="utf-8") as fdata_dice:
        fdata_dice.write(str(vocabulary_dict)+"\n")
    print(len(vocabulary_dict.keys()))
    print("分词结束！！")
    return vocabulary_dict


def word2index(sentcence,tokenizer_dist,max_sentencelength=350):
    #将句子转换为对应的索引
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
            else:  #这个步骤很关键，直接替换字典中key的值
                # print("当前未知词为：",word)
                updata_flag = True
                tokenizer_key_list = list(tokenizer_dist.keys())
                tokenizer_value_list = list(tokenizer_dist.values())
                # print(tokenizer_value_list)
                for index,key in enumerate(tokenizer_key_list):
                    if pattern.search(key):
                        unknow_index = index
                        # print(unknow_index)
                        break
                # return
                # unknow_index = tokenizer_value_list.index(2) #未知词对应的索引为2
                # print("初始化未知词索引",unknow_index)
                tokenizer_key_list[unknow_index]=word
                tokenizer_dist={key:tokenizer_key_list.index(key) for key in tokenizer_key_list}
                sentcence_lsit[temp_index]=(tokenizer_dist[word])
        temp_index+=1
    if updata_flag :
        # print("词表已经更新")
        updatatokenizer(tokenizer_dist)
        # print("词表更新结束！！")
    # print("需要转换的序列为：",sentcence)
    # print("序列转换后的结果为：",sentcence_lsit)
    # print(len(sentcence_lsit))
    return sentcence_lsit

def loadtokenizer(token_path):
    """
    实现从指定文件中读取数据，构成字典。
    """
    with open(token_path, 'r', encoding="utf-8") as fdata:
       tokenizer_dist = fdata.readlines()[-1].strip("\n")
    return ast.literal_eval(tokenizer_dist)


def updatatokenizer(tokenizer_dist):
    """
    对传入的tokenizer_dist进行处理，并且更新到最新字典记录文件中
    """
    with open("./tokenize_dict.txt","w",encoding="utf-8") as ftoken:
        ftoken.write(str(tokenizer_dist)+"\n")
    # print("字典数据更新结束！！")


if __name__ == '__main__':
    # file_path=r"../TransformerModel/trainsformer_datast.txt"
    # file_path=r"D:\Difference_comparison_vulnerability_detection_system\DetectMModel\loacal_tokennize\train_data.txt"
    # tokenizer_dist=vocabularygenerate(file_path)



    tokenizer_dist=loadtokenizer(r".\tokenize_dict.txt")
    # print("词表大小为：",len(list(tokenizer_dist)))
    stence="cmp dword ptr [0x423088], 0;je 0x4018f9;mov dword ptr [ebp - 4], 0;cmp dword ptr [0x423088], 0;je 0x40191f "
    result=word2index(stence,tokenizer_dist,200)
    print(result)
    # print(len(list(loadtokenizer(r"D:\graduation_project\loacal_tokennize\tokenize_dict1.txt"))))
    # loadtokenizer("./tokenize_dict1.txt")