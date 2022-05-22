import os 

def getfilename(path):
    return path.split('/')[-1]

def readtextfile(path):
    with open(path, 'r') as f:
        text = f.read()
    text = text.split('\n')
    text = strip_and_remove_empty(text)
    return text

def readfoldertext(folder, force=False):
    paths = os.listdir(folder)
    for path in paths:
        if force or path.endswith('.txt'):
            yield {path: readfoldertext(os.path.join(folder, path))}

def split_line(text, dlim=' '):
    return strip_and_remove_empty(text.split(dlim))

def strip_and_remove_empty(text_list):
    text = list(map(str.strip, text_list))
    text = list(filter(lambda x: x, text))
    return text

# number of unique gene covered by each func
