import os.path
import shutil
import torch.utils.data as Data
from fontTools.misc.psOperators import ps_integer
from tqdm import tqdm
import torch
import random
try:
    from  LocalTokenizer import word2index,loadtokenizer
except:
    from TransformerModel.LocalTokenizer import word2index, loadtokenizer

"""
# @Time    : 2025/07/17
# @Author  : huqinsong
# @Version: 数据预处理
# @Desc: 主要实现：
        将提取到缺陷汇编语句自动划分为Transformer可训练的数据集，并序列化汇编语句
"""


def generate_model_data(defetcode_setfile):
    """
    对生成的缺陷代码数据文件，再次进行预处理，使其能够适应深度学习的训练
    defetcode_setfile:缺陷汇编代码集文件
    """
    with open(defetcode_setfile,"r",encoding="utf-8") as fdefect:
        defectcode_content = fdefect.read()
    with open("./TransformerDefectcodeSet/trainsformer_datast.txt","w",encoding="utf-8") as ftrans:
        for codeline in defectcode_content.split("\n"):
            # print("huqinsong",codeline)
            # codeline=codeline.replace(";"," ; ")
            vultype = codeline.split("\t\t")[-1]
            defetcode_content = " ; ".join(codeline.split("\t\t")[:-1])
            # print(f"缺陷类型为{vultype},缺陷切片为{defetcode_content}")
            ftrans.write(defetcode_content+f" ! S {vultype} " +f"! {vultype} E" + "\n")
            ftrans.flush()


def split_data_and_getvocbsize(data_path,getvobsize=False):
    """
    划分训练集和测试集
    data_path:训练数据集的路劲
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

def count_max_seq_len(data_path):
    datas = open(data_path, 'r', encoding="utf-8").readlines()
    max_len = 0
    max_seq_sentence=None
    # for defactcode in tqdm(datas,desc="正在统计序列最大长度..."):
    for defactcode in datas:
        if max_len <= len(defactcode.split(" ")):
            max_len = max(max_len, len(defactcode.split(" ")))
            max_seq_sentence=defactcode
    # print("数据集中序列的最大长度为：",max_len)
    #print("当前的最大序列检查语句为:\n",max_seq_sentence)
    return max_len+10



class DefactcodeDataset(Data.Dataset):
    def __init__(self,tokenizerdict_path,data_path, max_seq_len=200):
        super().__init__()
        self.tokenizerdict_path=tokenizerdict_path
        self.tokenizer_dict =loadtokenizer(self.tokenizerdict_path)
        # print("数据预处理中读取到字典为：", self.tokenizer_dict)
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
        # print("encoder的输入为:", enc_input.strip())
        # print("dec_input的内容为:", dec_input.strip(), dec_input)
        # print("dec_output的内容为:", dec_output.strip(), dec_output)
        enc_input = word2index(enc_input,self.tokenizer_dict,self.max_seq_len)
        dec_input = word2index(dec_input.strip(" "), self.tokenizer_dict,self.max_seq_len)
        dec_output = word2index(dec_output.strip(" "),self.tokenizer_dict,self.max_seq_len)
        #self.tokenizer_dict = loadtokenizer(self.tokenizerdict_path)
        # print("正确的预测结果:",dec_output)
        self.data_cache[index] = (torch.LongTensor(enc_input), torch.LongTensor(dec_input), torch.LongTensor(dec_output))
        return torch.LongTensor(enc_input), torch.LongTensor(dec_input), torch.LongTensor(dec_output)


if __name__ == "__main__":

    #生成训练数据
    #generate_model_data(r"./DefectcodeSet/defectcode_set.txt")

    #划分数据集
    #defacetcode_set=r"./TransformerDefectcodeSet/trainsformer_datast.txt"
    #split_data_and_getvocbsize(defacetcode_set)
    # 计算整个数据集中序列的最大长度，并且创建分词对象
    # data_set=r".\trainsformer_datast.txt"
    data_set="./data/train.txt"
    tokenizer_dict=loadtokenizer(r"tokenize_dict.txt")
    max_seq_len=count_max_seq_len(data_set) # CWE416中对应的序列最大长度为774
    #print("当前最大序列长度为:",max_seq_len)
    # tokenizer_dict_path="./tokenize_dict1.txt"
    # dataset = DefactcodeDataset(tokenizer_dict_path, "./data/train.txt", max_seq_len)
    # print("tiaoshi",dataset[0])





