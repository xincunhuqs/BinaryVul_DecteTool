import os.path
import shutil

import torch.utils.data as Data
from tqdm import tqdm
import torch
import random
from  DetectMModel.loacal_tokennize.LocalTokenizer import word2index,loadtokenizer


def generate_model_data(defetcode_setfile):
    """
    对生成的缺陷代码数据文件，再次进行预处理，使其能够适应深度学习的训练
    defetcode_setfile:缺陷汇编代码集合文件
    """
    with open(defetcode_setfile,"r",encoding="utf-8") as fdefect:
        defectcode_content = fdefect.read()
    with open("./trainsformer_datast.txt","w",encoding="utf-8") as ftrans:
        for codeline in defectcode_content.split("\n"):
            vultype = codeline.split("\t\t")[-1]
            defetcode_content = " ; ".join(codeline.split("\t\t")[:-1])
            print(f"缺陷类型为{vultype},缺陷切片为{defetcode_content}")
            ftrans.write(defetcode_content+f" ! S {vultype} " +f"! {vultype} E" + "\n")
            ftrans.flush()


def split_data_and_getvocbsize(data_path,getvobsize=False):
    """
    划分训练集和测试集
    """
    file_datas = open(data_path, 'r', encoding="utf-8").readlines()
    data_sections=[]
    for data_section in tqdm(file_datas,desc="正划分数据集..."):
        data_sections.append(data_section)
    random.shuffle(data_sections)
    train_data = data_sections[:int(len(data_sections)*0.95)]
    test_data = data_sections[int(len(data_sections)*0.95):]
    if os.path.exists("./data/"):
        shutil.rmtree("./data/")
    if not os.path.exists("./data/"):
        os.makedirs("./data/")
    if not getvobsize:
        open("./data/train.txt", 'w', encoding="utf-8").writelines(train_data)
        open("./data/test.txt", 'w', encoding="utf-8").writelines(test_data)


def count_max_seq_len(data_path,tokenizer_dict):
    datas = open(data_path, 'r', encoding="utf-8").readlines()
    max_len = 0
    for defactcode in tqdm(datas,desc="正在统计序列最大长度..."):
        max_len = max(max_len, len(defactcode.split(" ")))
    # print("数据集中序列的最大长度为：",max_len)
    return max_len+100


class DefactcodeDataset(Data.Dataset):
    def __init__(self,tokenizerdict_path,data_path, max_seq_len=200):
        super().__init__()
        self.tokenizer_dict =loadtokenizer(tokenizerdict_path)
        self.datas = open(data_path, 'r', encoding="utf-8").readlines()
        self.max_seq_len = max_seq_len
        self.data_cache = {}

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        if index in self.data_cache:
            return self.data_cache[index]
        enc_input, dec_input, dec_output = self.datas[index].strip().split("!")
        # print(self.datas[index].strip().split("!"))
        print("encoder的输入为:", enc_input.strip())
        # print("test",len(enc_input.split()))
        print("dec_input的内容为:", dec_input.strip(), dec_input)
        print("dec_output的内容为:", dec_output.strip(), dec_output)
        enc_input = word2index(enc_input,self.tokenizer_dict,self.max_seq_len)
        dec_input = word2index(dec_input.strip(" "), self.tokenizer_dict,self.max_seq_len)
        dec_output = word2index(dec_output.strip(" "),self.tokenizer_dict,self.max_seq_len)
        self.data_cache[index] = (torch.LongTensor(enc_input), torch.LongTensor(dec_input), torch.LongTensor(dec_output))
        return torch.LongTensor(enc_input), torch.LongTensor(dec_input), torch.LongTensor(dec_output)


if __name__ == "__main__":

    #生成训练数据
    generate_model_data("./defectcode_set.txt")

    #划分数据集
    defacetcode_set=r"./trainsformer_datast.txt"
    split_data_and_getvocbsize(defacetcode_set)

    # 计算整个数据集中序列的最大长度，并且创建分词对象
    # data_set=r".\trainsformer_datast.txt"
    # tokenizer_dict=loadtokenizer(r"tokenize_dict.txt")
    # max_seq_len=count_max_seq_len(data_set,tokenizer_dict) # CWE416中对应的序列最大长度为774

    # tokenizer_dict_path="./tokenize_dict1.txt"
    # dataset = DefactcodeDataset(tokenizer_dict_path, "./data/train.txt", max_seq_len)
    # print("tiaoshi",dataset[0])





