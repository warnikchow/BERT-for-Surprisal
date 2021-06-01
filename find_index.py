prepos = ['with', 'in', 'at', 'by', 'outside', 'from', 'along', 'down', 'across', 'on', 'up']

def find_diff(arr1,arr2):
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            return i, arr2[i]

def find_phrase_start(tokens):
    for i in range(len(tokens)):
        if tokens[i] in ['had', 'was', 'were']:
            return i

phrend = ['ed', 'ng', 'pt', 'ot', 'en', 'wn', 'it', 'ug', 'ht', 'nk', 'ld', 'um', 'un', 'at', 'et',
          '#d', '#g', '#t', '#n', '#k', '#m']

def find_phrase_end(tokens):
    start = find_phrase_start(tokens)
    verb_start = start + 1
    for i in range(len(tokens)):
        if '#' not in tokens[verb_start+i+1]:
            return verb_start+i

def find_prepos_start(tokens):
    for i in range(len(tokens)):
        if tokens[i] in prepos:
            return i

def find_original_mask(tokens):
    for i in range(len(tokens)):
        if tokens[i] == '[MASK]':
            return i

def mask_start_end(arr,start,end):
    temp = []
    for i in range(len(arr)):
        temp.append(arr[i])
    if start != end:
        for i in (start,end):
            temp[i] = '[MASK]'
    else:
        temp[start] = '[MASK]'
    for i in range(5):
        if temp[i] in ['fire', 'weight', '##lift']:
            temp[i] = '[MASK]'
    return temp

def mask_start_end_pdep(arr,start,end):
    temp = []
    for i in range(len(arr)):
        temp.append(arr[i])
    if start != end:
        for i in (start,end):
            temp[i] = '[MASK]'
    else:
        temp[start] = '[MASK]'
    return temp

