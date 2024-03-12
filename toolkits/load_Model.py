'''
@File    :   load_Model.py
@Time    :   2024/03/12 09:47:07
@Author  :   shenlei ,annotated by yibozhao
@Version :   1.0
@Original:   https://github.com/netease-youdao/BCEmbedding/blob/master/BCEmbedding/models/embedding.py
@Site    :   https://github.com/zhaoyib/Job-Person-Matching
@description:an annotated edition of BCEmbedding produced by netease youdao.
'''
import torch
from tqdm import tqdm
from numpy import ndarray
from typing import List,Dict,Tuple,Type,Union
from transformers import AutoModel,AutoTokenizer
from logger import logger_wrapper
logger = logger_wrapper()
access_token = "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"

class Embedding:
    def __init__(self,model_name:str = 'maidalun1020/bce-embedding-base_v1',
                 pooler:str = 'cls',device:str = None, use_fp16:bool = False,
                 **kwargs) -> None:
        '''
        model name is auto set BCEmbedding_base_v1, you can change it as your need
        pooler is cls for default, which has been proved that it is better than mean in this model
        device will be detacted automatically.
        '''
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,token=access_token)
        self.model = AutoModel.from_pretrained(model_name,**kwargs, token = access_token)
        logger.info(f"loading from {model_name}")
        # load the pretrained model from hugging face.

        assert pooler in ['cls','mean'],f"pooler should be cls or mean, cls recommended"
        self.pooler = pooler
        #set the pooler, using assert to make sure it is in the list.

        #detact GPU devices. no need to change it.
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
        #end of detact and set GPU devices.

        #using fp16, which can reduce the using of VRAM
        if use_fp16:
            self.model.half()

        #using evaluating type, which will ban the dropout and batchnormalization in feed forward.
        #it can ensure the robustness of the model's output.
        self.model.eval()
        self.model = self.model.to(self.device)#deploy the model to GPU(if there is a GPU)

        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)#multi-GPU, using Parallel method.
        
        logger.info(f"Execute device: {self.device};\t gpu num: {self.num_gpus};\t use fp16: {use_fp16};\t embedding pooling type: {self.pooler};\t trust remote code: {kwargs.get('trust_remote_code', False)}")


    def encode(self,sentences:Union[str,list[str]],
               batch_size:int = 256,max_length:int = 512,
               normalize_to_unit:bool = True,return_numpy:bool = True,
               enable_tqdm:bool = True,query_instruction:str = "",
               **kwargs):
        '''
        sentences, which should be str or a list of str, as input.
        batch_size, default 256.
        max length can't be changed, default 512, if overflow, it will be truncated.
        normalize to unit means to normalize the embedding to gauss distribution.
        for bce-embedding-base_v1, keep query_instruction as "" is ok
        '''
        if self.num_gpus>1 :
            batch_size = batch_size * self.num_gpus
        
        if isinstance(sentences, str):
            sentences = [sentences]
            #unify the type of data.

        with torch.no_grad():# set as no grad, which won't update the weight of model.
            embeddings_collection = []
            for sentence_id in tqdm(range(0, len(sentences), batch_size), desc='Extract embeddings', disable=not enable_tqdm):
                #it will process all the sentence to batches.
                #and then process it. how many times it executes depends on the batch size and the count of sentence.
                if isinstance(query_instruction,str) and len(query_instruction)>0:
                    sentence_batch = [query_instruction + sent for sent in sentences[sentence_id:sentence_id+batch_size]]
                else:
                    sentence_batch = sentences[sentence_id:sentence_id+batch_size]
                #build batch.
                    
                inputs = self.tokenizer(
                    sentence_batch,
                    padding = True,
                    truncation = True,
                    max_length = max_length,
                    return_tensors = "pt"
                )
                inputs_on_device = {k:v.to(self.device) for k,v in inputs.items()}
                #inputs on device include two parts: input_ids, which is a tensor including the token ids.0 as cls,1 as sep
                #another is the attention_mask. 1 as lookable, 0 as mask
                outputs = self.model(**inputs_on_device,return_dict = True)

                if self.pooler == 'cls':
                    embeddings = outputs.last_hidden_state[:,0]#it will be tensor with shape [batch_size,768]
                    #which means all the meaning will be collected to the first token 'cls'
                elif self.pooler == 'mean':
                    attention_mask = inputs_on_device['attention_mask']
                    last_hidden = outputs.last_hidden_state# it will be tensor with shape [batch_size,length,768]
                    embeddings = ((last_hidden * attention_mask.unsqueeze(-1).float()).sum(1)
                                  / attention_mask.sum(-1).unsqueeze(-1))
                else:
                    raise NotImplementedError
                
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim = 1, keepdim = True)
                embeddings_collection.append(embeddings.cpu())

            embeddings = torch.cat(embeddings_collection,dim=0)
            #embedding_collection with shape: [len(sentences)/batch_size,batch_size,768]
            #it will be cat to [len(sentences),768]

        if return_numpy and not isinstance(embeddings,ndarray):
            embeddings = embeddings.numpy()
            #change it to numpy.ndarray
        return embeddings
    
if __name__ == '__main__':
    model = Embedding()
    inputs_on_device = {k:v.to("cuda:0") for k,v in model.tokenizer(['调试由天下人才公司Peter完成','欢迎访问他的github','他的用户名是zhaoyib']
                                                                    ,padding = True,truncation = True,return_tensors = "pt").items()}
    print(inputs_on_device)
    outputs = model.model(**inputs_on_device,return_dict = True)
    print(outputs.last_hidden_state[:,0].shape)