import importlib.util
import subprocess
import sys,os,time


"""
# @Time    : 2025/08/18
# @Author  : huqinsong
# @Version: 预安装python模块
# @Desc: 主要实现：
        在执行二进制文件扫描之前先检查必备的python模块是否安装
"""

def necemodel_preinstall():
    """
    运行该工具时自动检查并安装所需的模块

    """
    write_time=time.strftime("%Y-%m-%d_%H:%M:S",time.localtime())
    necessary_model_dict={"numpy":"2.2.0","torch":"2.5.1","openai":"1.58.1","capstone":"5.0.3","pefile":"2024.8.26",\
                         "click":"8.1.7","tqdm":"4.67.1" }
    print(f"\033[31m请稍作等待，正在检查前置包...\033[0m")
    preinstall_error=""
    checklog_path=f"./TransformerModel/check_log.txt"
    with open(checklog_path,"w",encoding="utf-8") as fcheck:
        fcheck.write(f"*************{write_time}********************\n")
        for module_name,version in necessary_model_dict.items():
            install_flage=importlib.util.find_spec(module_name)
            if install_flage:
                print(f"\033[32m{module_name} 模块已安装...\033[0m")
                fcheck.write(f"{module_name}=={version} 模块已安装...\n")
            else:
                print(f"\033[31m正在安装:{module_name}模块.....\033[0m")
                fcheck.write(f"正在安装:{module_name}=={version}模块.....\n")
                try:
                    python_path=sys.executable
                    subprocess.check_call([f"{python_path}","-m","pip","install",f"{module_name}==={version}",
                                        "-i","https://mirrors.aliyun.com/pypi/simple",])
                    print(f"\033[32m{module_name}模块已安装....\033[0m")
                    fcheck.write(f"{module_name}=={version}模块已安装....\n")
                except subprocess.CalledProcessError as error:
                    print(f'\033[31m安装出错:\n {error} \033[0m')
                    fcheck.write(f'安装出错:\n {error} ')
                    preinstall_error=error.returncode
    if preinstall_error:
        os.remove(checklog_path)
        return 
    print(f"\033[31m当前环境检测结束....\033[0m")


if __name__=="__main__":
    necemodel_preinstall()
