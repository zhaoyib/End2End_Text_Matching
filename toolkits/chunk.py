'''
@File      :   chunk.py
@Time      :   2024/03/13 11:19:39
@LastEdit  :   2024/03/13 15:15:10
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
'''
import re
from typing import List,Union
from transformers import AutoTokenizer
from toolkits.logger import logger_wrapper
logger = logger_wrapper()
model_name = 'maidalun1020/bce-embedding-base_v1'
access_token = "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"

class Chunker:
    def __init__(self,method:dict = {"mode":"sliding_window",
                                    "window_size":300,"overlap":50}) -> None:
        """
        Initializes the Chunker with a specified strategy.

        parameter: method      : a dictionary containing the chunking mode (paragraph or sliding_window)
                                and optional window_size and overlap values for sliding_window mode.
                                {"mode":"sliding_window","window_size":300,"overlap":60} as default
                   mode        : sliding_window or paragraph.
                   window_size : an int less than 350.
                   overlap     : the length of overlap,if it is int, less than window size; if float, 0-1.
        return   : None
        """
        self.mode = method['mode']
        assert self.mode in ["sliding_window","paragraph"], \
            "invalid mode chosen, please reset it to sliding window or paragraph"
        self.window_size = method["window_size"]
        self.overlap = method["overlap"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,token=access_token)
        logger.info(f"Init the Chunker successfully, tokenizer from {model_name}")
        pass

    @staticmethod
    def cleasing(text:str)->str:
        '''
        Remove extra whitespace with single whitespace.

        parameter:  text : a string containing the text to be cleaned.
        return   :  a cleaned version of the input text.  
        '''
        text = re.sub(r"\s+", " ", text).strip()
        #Warning. DON'T Remove ALL the Whitespace. It Will Lead to an IRREVERSIBLE Result. The English Text Will Be Unreadable. 
        return text
    
    def __call__(self, inputs: dict) -> List[str]:
        '''
        once a text, return a dict
        '''
        id = str(list(inputs.keys())[0])
        text = list(inputs.values())
        if self.mode == "sliding_window":
            text_list = self.sliding_window_chunking(text[0])
            #print(text_list)
            res = {}
            for index, part_of_text in enumerate(text_list):
                final_id = id + '_' + str(index)
                new = {final_id:part_of_text}
                #print(new)
                res.update(new)
            return res
        else:
            text_list = self.paragraph_chunking(text[0])
            res = {}
            for index, part_of_text in enumerate(text_list):
                final_id = id + '_' + str(index)
                res.update({final_id:part_of_text})
            return res
        
    

    def paragraph_chunking(self, text:str) -> List[str]:
        '''
        Split the input text into paragrahps.

        parameter: text : a string including the raw text.
        return   : a list of chunked text.
        '''
        paragraphs = text.split("\n")# It should be "\n\n" in English contents, but we can use it as well.
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            cleaned_paragraph = self.cleasing(paragraph)
            if len(cleaned_paragraph) > 1:#to filter the text with no word, set >1 to avoid the "\n" or otherwise.
                cleaned_paragraphs.append(cleaned_paragraph)
        return cleaned_paragraphs
    
    def sliding_window_chunking(self, text:str) -> List[str]:
        '''
        Split the input text into chunks using the sliding window method.

        parameter: text : a string including the raw text.
        return   : a list of chunked text
        '''
        assert isinstance(self.window_size,int), "Please Input A Valid Window Size Value, Int Only."
        assert self.window_size < 350, "Too Big Window Size, Please Limit It Less Than 350."
        assert isinstance(self.overlap,Union[int,float]), "Please Input A Valid Overlap Value, Int or Float."
        if isinstance(self.overlap,int):
            assert self.overlap < self.window_size, "Don't Set The Overlap More Than Window Size."
        if isinstance(self.overlap, float):
            assert self.overlap < 1, "The Percentage of Overlap Should Be Less Than 1"
            assert self.overlap >= 0, "The Percentage of Overlap Should Be NO Less Than 0"
            self.overlap = int(self.window_size * self.overlap)
        
        text = self.cleasing(text)
        tokens = self.tokenizer(text,padding=True,return_overflowing_tokens=True,
                                stride=self.overlap,max_length=self.window_size,
                                return_tensors = "pt",truncation=True)
        tokens["chunks"] = []
        for ids in tokens["input_ids"]:
            tokens["chunks"].append(self.tokenizer.decode(ids))
        #print(tokens)
        return tokens["chunks"]
