'''
@File      :   load_Reranker.py
@Time      :   2024/03/12 15:59:08
@LastEdit  :   2024/03/19 10:12:50
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
'''
import logging
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Type, Union
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from toolkits.logger import logger_wrapper
from toolkits.Reranker_preprocess import reranker_tokenizer_preproc

logger = logger_wrapper()
access_token = "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"

class Reranker:
    def __init__(self,model_name: str='maidalun1020/bce-reranker-base_v1',
                 use_fp16:bool = False, device:str = None, **kwargs) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,token=access_token)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name,**kwargs,token=access_token)
        logger.info(f"Loading from `{model_name}`.")

        num_gpus = torch.cuda.device_count()
        if device is None:
            self.device = "cuda" if num_gpus > 0 else "cpu"
        else:
            self.device = 'cuda:{}'.format(int(device)) if device.isdigit() else device
        
        if self.device == "cpu":
            self.num_gpus = 0
        elif self.device.startswith('cuda:') and num_gpus > 0:
            self.num_gpus = 1
        elif self.device == "cuda":
            self.num_gpus = num_gpus
        else:
            raise ValueError("Please input valid device: 'cpu', 'cuda', 'cuda:0', '0' !")

        if use_fp16:
            self.model.half()

        self.model.eval()
        self.model = self.model.to(self.device)

        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
        #like BCEmbedding, no difference.
        logger.info(f"Execute device: {self.device};\t gpu num: {self.num_gpus};\t use fp16: {use_fp16}")

        # for advanced preproc of tokenization
        self.max_length = kwargs.get('max_length', 512)#max length.
        self.overlap_tokens = kwargs.get('overlap_tokens', 80)
    
    def compute_score(self, sentence_pairs:Union[List[Tuple[str,str]],Tuple[str,str]],
                      batch_size:int = 256, max_length:int = 512, enable_tqdm:bool = True,
                      **kwargs):
        if self.num_gpus > 1 :
            batch_size = batch_size * self.num_gpus
        
        assert isinstance(sentence_pairs,list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]
        
        with torch.no_grad():
            score_collection = []
            for sentence_id in tqdm(range(0,len(sentence_pairs),batch_size),
                                    desc='Calculate Scores',disable= not enable_tqdm):
                sentence_pairs_batch = sentence_pairs[sentence_id:sentence_id+batch_size]
                inputs = self.tokenizer(
                            sentence_pairs_batch, 
                            padding=True,
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt"
                        )
                inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
                #all of the code above has no difference with Embedding.
                scores = self.model(**inputs_on_device,return_dict = True).logits.view(-1).float()
                scores = torch.sigmoid(scores)
                score_collection.extend(scores.cpu().numpy().tolist())
            
        if(len(score_collection) == 1):
            return score_collection[0]
        return score_collection
    
    def rerank(self, query:str, CVs:list, batch_size:int = 32,**kwargs):
        '''
        rerank the CVs.

        parameters:
            query   : the text of Job Define.
            CVs     : the list of conditate, elements are tuples, (cv_id_index, text, embedding, sim)
        return:
            'rerank_passages': sorted_passages,
            'rerank_scores'  : sorted_scores,
            'rerank_ids'     : sorted_cvids.
        '''
        passages = [item[1] for item in CVs]
        ids = [item[0] for item in CVs]

        if query is None or len(query)==0 or len(passages)==0:
            return {"rerank_passages":[],"rerank_scores":[]}

        #preprocess in Reranker_preprocess.py.
        sentence_pairs, sentence_pairs_pids = reranker_tokenizer_preproc(
            query, passages, 
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            overlap_tokens=self.overlap_tokens,
            )
        #sentence_pairs,a list of raw texts which merged both query and passage.
        #sentence_pairs_pids, a list of ids.
        
        #batch inference.
        if self.num_gpus > 1:
            batch_size = batch_size * self.num_gpus


        tot_scores = []
        with torch.no_grad():
            for k in range(0, len(sentence_pairs), batch_size):
                batch = self.tokenizer.pad(
                        sentence_pairs[k:k+batch_size],
                        padding=True,
                        max_length=None,
                        pad_to_multiple_of=None,
                        return_tensors="pt"
                    )
                #pad the input.
                batch_on_device = {k: v.to(self.device) for k, v in batch.items()}#put batch to device.
                scores = self.model(**batch_on_device, return_dict=True).logits.view(-1,).float()
                scores = torch.sigmoid(scores)
                tot_scores.extend(scores.cpu().numpy().tolist())
                #calculate the scores.
        
        # ranking
        merge_scores = [0 for _ in range(len(passages))]
        for pid, score in zip(sentence_pairs_pids, tot_scores):
            merge_scores[pid] = max(merge_scores[pid], score)

        merge_scores_argsort = np.argsort(merge_scores)[::-1]
        sorted_passages = []
        sorted_scores = []
        sorted_cvids = []
        for mid in merge_scores_argsort:
            sorted_scores.append(merge_scores[mid])
            sorted_passages.append(passages[mid])
            sorted_cvids.append(ids[mid])

        return {
            'rerank_passages': sorted_passages,
            'rerank_scores': sorted_scores,
            'rerank_ids': sorted_cvids
        }
    

if __name__ == "__main__":
    query = "天猫平台Gucci运营"
    passages = ["王**在职，看看新机会女24岁上海-杨浦区本科工作3年4k电商运营NIKE大中华区总部************立即沟通推荐职位查看联系方式收藏转发求职意向电商运营上海-杨浦区服装/纺织/皮革工作经历NIKE大中华区总部（2020.12-至今,3年3个月）服装/纺织/皮革电商运营所在部门：T-mallDigital薪资：4k职位类别：电商运营职责业绩：负责NIKE淘宝天猫旗舰店的电商运营工作，进行日常货品上下架选择及排墙顺序的安排、参与双十二、双旦、天猫年货节等大型天猫活动的事项跟进，对其他辅助部门下达活动指令。负责每周周会记录工作，同时对每周的生意进行汇总，跟进投入最多的产品销售情况，对竞品公司进行一定的分析，从而协助主管进行营销手段的预判和修正。",
                "2022/07-2022/11HR实习生?主要负责快消品、零售领域EC、Marketing等市场销售的中高阶职位招聘，包括头部美妆品牌以及日用品、食品品牌等，对top30快消品牌有较为熟悉的了解。?密切关注人才市场信息，进行人才地图绘制和企业定向mapping。?初步面试候选人，与客户方HR沟通以及跟进候选人面试教育经历上海海事大学·国际经济与贸易·本科2021.09-2025.06统招本科"]
    sentence_pairs = [[query,passage] for passage in passages]
    model = Reranker(device="cuda:0")
    scores = model.compute_score(sentence_pairs)
    print(scores)
    print(f"VMemory used for compute score {torch.cuda.max_memory_allocated()/1000000} Mb")
    reranker_result = model.rerank(query,passages)
    print(reranker_result)
    print(f"VMemory used for rerank {torch.cuda.max_memory_allocated()/1000000} Mb")
