# End2End Text Matching
### 项目结构
```
├─model
│  ├─__pycache__    # folder built automatically.
│  ├─__init__.py    # empty file to build module.
│  ├─Embeder.py     # embedder model class.
│  └─Reranker.py    # reranker model class.
├─toolkits
│  ├─__pycache__    # folder built automatically.
│  ├─__init__.py    # empty file to build module.
│  ├─chunk.py       # chunker class.
│  ├─logger.py      # record logger.
│  ├─Reranker_preprocess.py # util of reranker.
│  └─utils.py       # file reader and format output.
├─new_pipeline.py   # pipeline of the whole model.
├─requirements.txt  # requirements of the project.
└─README.md
```
### 项目依赖
为了能够成功运行本项目，请先安装python `3.10`，`3.10`以下的版本可能也可以运行，但是未经过测试。<br>
本项目依赖库已经导出到了requirements.txt文件(如果无法安装，请删除requirements.txt中每行最后一个'='以及之后的内容)，可以通过下面的命令安装：<br>
`pip install -r requirement.txt`<br>
同时需要安装Cuda12.1版本以适配本项目使用的GPU计算，如果无法使用GPU计算，则可以选择不安装，使用CPU以较慢速度计算。
### 声明
出于隐私方面考虑，项目中所涉及的一切数据均不会对外提供。模型仅供参考学习使用，禁止商用。
### 作者
Yibo Zhao, 联系方式: 10203330408@stu.ecnu.edu.cn
也可联系：zhaoyibo624@gmail.com
### 致谢
感谢有道开发的BCEmbedding以及BCEreranker模型以及相关开源代码。
[有道BCEmbedding Github项目地址](https://github.com/netease-youdao/BCEmbedding)
