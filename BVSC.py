
from TransformerModel.preinstall_model import necemodel_preinstall
import os.path,time
if not os.path.exists(f"./TransformerModel/check_log.txt"):
        necemodel_preinstall()

from openai import OpenAI
from tqdm import tqdm
from capstone import *
import pefile
import time
import click
from TransformerModel.transformer_v3 import defactcode_detect

"""
# @Time    : 2025/07/17
# @Author  : huqinsong
# @Version: 差异化比对二进制漏洞检测系统
# @Desc: 主要实现：
         对传入的二进制文件进行漏洞检测
"""


SLICING_BOCK_SIZE=80
def disassembly_exe(executable_file_path,save_folder=None):
    """
    传入指定的可执行二进制文件，将其进行反汇编，然后提取出.text节中的反汇编代码
    executable_file_path:需要编译的可执行二进制文件
    """

    pe = pefile.PE(executable_file_path)
    for item in pe.sections:
        #当从外部数据源读取数据时，数据通常以字节流的形式表示。
        if  ".text" in str(item.Name.decode('UTF-8')) :  #decode函数将字节流解码为字符串
            VirtualAddress = item.VirtualAddress  # 获取text节的相对虚拟地址
            VirtualSize = item.SizeOfRawData     # 获取text节的虚拟大小
            ActualOffset = item.PointerToRawData  #获取text节在文件中的偏移量
    ImageBase =pe.OPTIONAL_HEADER.ImageBase 
    StartVA =ImageBase +VirtualAddress  # 计算出.text节的虚拟地址

    with open(executable_file_path, "rb") as fexefile:
        fexefile.seek(ActualOffset,1)   #将文件指针调整到可执行文件的text节位置处，然后从该位置开始读取数据
        HexCode = fexefile.read(VirtualSize)  # 可执行文件中的text节内容全部进行读取

    filename = os.path.basename(executable_file_path).split(".")[0]
    disassembly_contents = ""
    md = Cs(CS_ARCH_X86, CS_MODE_32)  # 初始化Capstone引擎，指定x86架构和32位模式
    for item in md.disasm(HexCode, StartVA):  # 反汇编节中的代码
        disassembly_contents += str(hex(item.address)+" "+item.mnemonic + " " + item.op_str)+"\n"
    if save_folder:
        folder_disassembling = save_folder + "//disassembly_folder"
        if not os.path.exists(folder_disassembling):
            os.makedirs(folder_disassembling)
        assembly_file = os.path.join(folder_disassembling, filename + ".txt")
        with open(assembly_file,"w",encoding="utf-8") as fassemb:
            fassemb.truncate(0)
            fassemb.write(disassembly_contents)

    return disassembly_contents



def get_slicing(execute_file,bisc_size):
    """
    实现对可执行文件进行反汇编，并且进行切片
    execute_file:可执行文件路径
    bisc_size:切片代码的大小

    """
    slicing_number = 0
    base_dir = "./temp_folder"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    disassembly_linecontent = disassembly_exe(execute_file,)
    disassembly_file = os.path.join(base_dir, "temp_disassembly.txt")
    with open(disassembly_file,"w",encoding="utf-8") as fdisassem:
        fdisassem.write(disassembly_linecontent)
    disassembly_content_list = disassembly_linecontent.split("\n")
    slicing_filename = os.path.join(base_dir,"temp_slicing.txt")
    with open(slicing_filename,'w',encoding="utf-8") as slicingf:
        for index in range(0,len(disassembly_content_list),SLICING_BOCK_SIZE):
            slicing_number+=1
            slicingf.write(f"===========slicing number:{slicing_number}===========\n")
            for contentline in disassembly_content_list[index:index+bisc_size]:
                contentline_str =" ".join(contentline.split(" ")[1:])+";"
                slicingf.write(contentline_str)
            slicingf.write("\n")
    with open(slicing_filename, 'r', encoding="utf-8") as slicingf1:
        slicing_content = slicingf1.read().split("\n")
    return slicing_content



def defectcode_withdrow(exeute_file,verbose):
    """
    实现对缺陷代码进行提取
    execute_file: 可执行文件路径
    verbose: 是否展示预测相关参数
    """
    bisc_size = SLICING_BOCK_SIZE
    slicing_content = get_slicing(exeute_file, bisc_size)
    defectcode_result_dict = {}
    for source_line in tqdm(slicing_content,desc="请稍作等待，正在提取缺陷代码..."):
        if not  source_line.startswith("==="):
            #潜在bug,临时处理
            try:
                result = defactcode_detect(r"./TransformerModel/local_model/transformer.pth", source_line,verbose)
                if result is None:
                    continue
                if not isinstance(result, (tuple, list)) or len(result) != 2:
                    continue
                defecode, vultype = result
            except Exception as e:
                print(f"意外异常: {e}")
                continue
            if "CWE" in vultype and defecode and vultype:
                defectcode_result_dict[defecode] = vultype
    return defectcode_result_dict


def analyze_defective_code(vultype,defective_code):
    #对生成的汇编语句进行详细的解析
    client = OpenAI(api_key="sk-e0be04fc2a574262a433f03282717cad", base_url="https://api.deepseek.com")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个资深的二进制文件逆向分析漏洞检测安全专家"},
                {"role": "user", "content": f"请认真分析一下{defective_code}这段汇编代码中,确保存在{vultype}漏洞的真实性。给出分析过程\
                 回答格式为：\
                 准确且可利用/准确但不可利用/不准确， 原因如下："},
            ],
            stream=False
        )
        repose_result=response.choices[0].message.content.split("\n")
        analyze_result = []
        for analyze in repose_result:
            if analyze:
                analyze_result.append(analyze.replace("###",''))
        analyze_result = "\n".join(analyze_result)
        return analyze_result
    except Exception as error:
        print(f"deepseek 解析失败！{error}")


def detect_tool(exefile,verbose,rsd_flage=False,analyze_flage=False):
    """
    根据对传入的可执行文件进行解析，检测潜在缺陷
    exefile:待检测的二进制文件
    savelog_file:记录检测过程中的安全的反汇编缺陷代码存储在savelog_file文件中
    rsd_flage:是否保存安全反汇编代码的标志
    analyze_flage:是否调用deepseek api对分析结果进行降噪
    verbose:展示检测过程中的详细信息
    """
    print(f"\n\033[32m------------------当前待检测的二进制文件----------------\033[0m")
    print(f"\033[32m{exefile}\033[0m")
    print(f"\033[32m-------------------------------------------------------\033[0m\n")
    defectcode_result_dict = defectcode_withdrow(exefile,verbose)
    result_dic={}
    if not os.path.exists("./DefectDiscoveryTrainDate"):
        os.mkdir("./DefectDiscoveryTrainDate")
    local_time=time.strftime("%Y-%m-%d_%H:%M:%S",time.localtime())
    savelog_file=rf"./DefectDiscoveryTrainDate/defectdate_discovery_{local_time}.txt"

    with open("./singchecked_result.txt",'a',encoding="utf-8") as fresult:
        defectcode_index=1
        for defect_code,vultype in tqdm(defectcode_result_dict.items(),desc="正在核验分析结果,请稍作等待..."):
            if analyze_flage:
                analyze_result = analyze_defective_code(vultype,defect_code)
                if analyze_result:
                    if "准确且可利用" in analyze_result:
                        analyze_result=analyze_result.replace('\\n','')
                        result_dic[defect_code]=vultype + f"--{analyze_result}"
                        defect_code=';\n'.join(defect_code.split(';'))
                        fresult.write(f"\n--------------------------------------缺陷代码块索引：{defectcode_index}-------------------------------------------------------\n")
                        fresult.write(f"检测时间：{local_time} \n检测文件:{exefile}\n检测结果:{vultype}\n可疑缺陷汇编代码块:{defect_code}\n：\n分析结果：{analyze_result}\n")
                        fresult.write(f"--------------------------------------------------------------\n")
                        defectcode_index+=1
            elif analyze_flage == False:
                    defect_code=';\n'.join(defect_code.split(';'))
                    fresult.write(f"\n--------------------------------------缺陷代码块索引：{defectcode_index}-------------------------------------------------------\n")
                    fresult.write(f"检测时间：{local_time} \n检测文件:{exefile}\n检测结果:{vultype}\n可疑缺陷汇编代码块:\n{defect_code}：\n")
                    fresult.write(f"--------------------------------------------------------------\n")
                    defectcode_index+=1
            else:
                if rsd_flage:
                    with open(savelog_file,'w+',encoding="utf-8") as fdiscover:
                        discover_content=fdiscover.read()
                        if defect_code not in discover_content:
                            fdiscover.write(defect_code+"  security_code"+"\n")
    # if os.path.exists(r"./tokenize_dict.txt"):
    #     os.remove(r"./tokenize_dict.txt")
    print(f"\033[32m 检测结束一共检测到:{defectcode_index}个可疑缺陷汇编代码块\033[0m")
    print(f"\033[31m 检测结果见：single_deteresult.txt \033[0m")


def batch_detect(folder_path,verbose):
    """
    接收可执行文件所在的文件夹，批量化检测二进制文件
    # """
    exefilepath_list=os.listdir(folder_path)
    print(f"\n********待检测的二进制文件列表信息如下:************\n",)
    for exefile in exefilepath_list:
        if ".exe" in exefile:
            print(exefile)
    print("\n*************************************************\n",)

    for exefilename in exefilepath_list:
        if ".exe" in exefilename:
            exefile_path=os.path.join(folder_path,exefilename)
            detect_tool(exefile_path,verbose)




@click.command()
@click.option("-efp","--exefile_path",help="Input the path of the binary file for detection",show_default=True)
@click.option("-efdp","--exefile_folder_path",help="Input the folder of the binary file for detection",show_default=True)
@click.option("-nrsc","--normal_scan",default=True,help="Normal scan model",show_default=True)
@click.option("-acsc","--accurate_scan",default=False,help="Combined with deepseek to do preliminary analysis" \
" to reduce false positives",show_default=True)
@click.option("-v","--verbose",default=False,help="Display detailed diagnostics during vulnerability assessment",show_default=True)
# @click.option("-rsd","--recordsecure_disassembly",default=False,type=bool,help="Record secure defect assembly code snippets during detection",show_default=True)
# @click.option("-vv","--")

def exefile_check(exefile_path,exefile_folder_path,normal_scan,accurate_scan,verbose):
    """
    对可疑二进制文件进行检测
    """
    if exefile_path:
        if accurate_scan == "True":
            print(f"\033[31m调用大模型对预测结果进行降噪\033[0m")
            detect_tool(exefile_path,verbose,rsd_flage=True,analyze_flage=True)
        elif normal_scan:
            print(f"\033[31m普通扫描二进制漏洞检测模式\033[0m")
            detect_tool(exefile_path,verbose,rsd_flage=False,analyze_flage=False)
    if exefile_folder_path:
        print(f"\033[32m当前检测的文件夹为:{exefile_folder_path}\033[0m")
        if accurate_scan == "True":
            print(f"\033[31m调用大模型对预测结果进行降噪\033[0m")
            batch_detect(exefile_folder_path,verbose)
        elif normal_scan:
            print(f"\033[31m普通扫描二进制漏洞检测模式\033[0m")
            batch_detect(exefile_folder_path,verbose)



# @click.command()
# @click.option("-rsd","--recordsecure_disassembly",default=False,type=bool,help="Record secure defect assembly code snippets during detection",show_default=True)
# @click.option("-rtm","--retrain_model",help="The code slice optimization model recorded during the detection process")
# def retrain_model():
#     pass

if __name__=="__main__":
        exefile_check()
