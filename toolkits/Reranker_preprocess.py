'''
@File      :   Reranker_preprocess.py
@Time      :   2024/03/12 17:34:51
@LastEdit  :   2024/03/19 09:14:56
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
'''
from typing import List, Dict, Tuple, Type, Union
from copy import deepcopy

def reranker_tokenizer_preproc(
        query:str, passages:List[str], tokenizer = None,
        max_length:int = 512, overlap_tokens:int = 80
        ):
    '''
    process the query and condidates.input query as str and passages as list of str.
    output the texts merging query and each passage. raw text and ids.

    parameter:
        query     : str, no longer than 200 tokens.
        passages  : list of str, which means the Texts chosen by rough retrieval.
        tokenizer : the tokenizer loaded from pretrained model.
        max_length: the max input length of model.
        overlap_tokens: how may tokens to overlap. 
    return:
        res_merge_inputs      : the list of merged text.
        res_merge_inputs_pids : the list of merged text, ids instead of raw text.
    '''
    assert tokenizer is not None, "Please Provide A Valid Tokenizer for Tokenization!"
    sep_id = tokenizer.sep_token_id #get the id of <sep>

    def _merge_inputs(chunk1_raw, chunk2):
        chunk1 = deepcopy(chunk1_raw)
        chunk1['input_ids'].extend(chunk2['input_ids'])
        chunk1['input_ids'].append(sep_id)
        chunk1['attention_mask'].extend(chunk2['attention_mask'])
        chunk1['attention_mask'].append(chunk2['attention_mask'][0])
        if 'token_type_ids' in chunk1:
            token_type_ids = [1 for _ in range(len(chunk2['token_type_ids'])+1)]
            chunk1['token_type_ids'].extend(token_type_ids)
        return chunk1
    
    query_inputs = tokenizer.encode_plus(query, truncation=False, padding=False)
    #print("query inputs from reranker preprocess",query_inputs)
    #->{'input_ids':……,'attention_mask':……}
    max_passage_inputs_length = max_length - len(query_inputs['input_ids']) - 1
    #max length - length of query and len of one simple seq token(mentioned at line 16).
    assert max_passage_inputs_length > 100, "Your query is too long! Please make sure your query less than 400 tokens!"
    #512 tokens in total. can't allow the query more than 400.
    overlap_tokens_implt = min(overlap_tokens, max_passage_inputs_length//4)

    res_merge_inputs = []
    res_merge_inputs_pids = []
    for pid, passage in enumerate(passages):
        passage_inputs = tokenizer.encode_plus(passage, truncation=False, padding=False, add_special_tokens=False)
        passage_inputs_length = len(passage_inputs['input_ids'])

        if passage_inputs_length <= max_passage_inputs_length:
            qp_merge_inputs = _merge_inputs(query_inputs, passage_inputs)
            res_merge_inputs.append(qp_merge_inputs)
            res_merge_inputs_pids.append(pid)
        else:
            start_id = 0
            while start_id < passage_inputs_length:
                end_id = start_id + max_passage_inputs_length
                sub_passage_inputs = {k:v[start_id:end_id] for k,v in passage_inputs.items()}
                start_id = end_id - overlap_tokens_implt if end_id < passage_inputs_length else end_id

                qp_merge_inputs = _merge_inputs(query_inputs, sub_passage_inputs)
                res_merge_inputs.append(qp_merge_inputs)
                res_merge_inputs_pids.append(pid)
    
    return res_merge_inputs, res_merge_inputs_pids
