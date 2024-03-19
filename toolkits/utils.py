import os

def file_reader(folder_path,batch_size = 400):
    '''
    yeild the file text list.

    parameter: folder_path: the path of the files' folder.
               batch_size : how many files visit once.
    WARNING  : To Avoid Being Out of Memory, Don't Set the Batch Size TOO BIG!
    return   : A dict of the Text, the key is the cv_id, the value is the texts. 
    '''
    items = os.listdir(folder_path)
    total_files = len(items)
    for i in range(total_files // batch_size + 1):
        start_index = i * batch_size
        end_index = min(start_index + batch_size, total_files)
        res = {}
        for j in range(start_index, end_index):
            item_path = os.path.join(folder_path, items[j])
            with open(item_path,'r') as f:
                content = f.read()
            cv_id = items[j].split('_')[1].split('.')[0]
            res.update({cv_id:content})
        if res:
            yield res
