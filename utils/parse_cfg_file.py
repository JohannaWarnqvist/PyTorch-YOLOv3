def parse_cfg_file(cfg_file_path):
    """ Function that parses the yolov3.cfg file so the information can be used to create pytorch modueles.
    
    Args: cfg_file path is the path to the yolov3.cfg file as a string. 
    
    Returns: A list containing dictionaries, each dictionary contains information on how to create the modules.
    """
    file = open(cfg_file_path, 'r') # read csv file
    lines = file.read().split('\n') #slit file on new line and create a list

    block_list = [] 
    block = {}
    counter = 0
    for line in lines:
        if len(line) < 1:
            #remove empty lines
            pass 
        
        elif line[0] == '#':
            #remove comments
            pass
       
        elif line[0] == '[': #first character of a block
            
            if len(block) > 0:
                block_list.append(block)
                block = {}
            block['type'] = line[1:-1]
            if block['type'] =='yolo':
                counter_yolo = counter
            counter += 1
            
        elif '=' in line:
            info = line.split('=')
            block[info[0].rstrip()] = info[1].rstrip()
            
    block_list.append(block)
    block = {}
                
    return block_list
