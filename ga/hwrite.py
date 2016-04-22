# Hwrite.py
# Brandon Jones, Jonathan Roberts, Josiah Wong
# Created April 8, 2016
# Homework 4

import math

pad = "                                                 " # Hi!


""" INTEGER AND DOUBLE WRITING METHODS """

def left(data, field_length, output):
    """ Write left justified integer/double """
    
    data_text = str(data)
    if field_length < len(data_text):
        field_length_error(field_length, output)
        return
    
    padding = pad[0:(field_length - len(data_text))]
    output.write(data_text + padding)
    
    
def right(data, field_length, output):
    """ Write right justified integer/double """
    
    data_text = str(data)
    if field_length < len(data_text):
        field_length_error(field_length, output)
        return
    
    padding = pad[0:(field_length - len(data_text))]
    output.write(padding + data_text)
    

def center(data, field_length, output):
    """ Write centered integer/double """
    
    data_text = str(data)
    if field_length < len(data_text):
        field_length_error(field_length, output)
        return
    
    total_pad_length = field_length - len(data_text)
    end_pad_length = int(total_pad_length / 2)
    start_pad_length = total_pad_length - end_pad_length
    
    start_padding = pad[0:start_pad_length]
    end_padding = pad[0:end_pad_length]
    output.write(start_padding + data_text + end_padding)
    
    
def left_places(data, field_length, places, output):
    """ Write left justified double - user specified decimal places """
    
    data_text = fix_decimals(data, places)
    if field_length < len(data_text):
        field_length_error(field_length, output)
        return
    
    padding = pad[0:(field_length - len(data_text))]
    output.write(data_text + padding)
    
    
def right_places(data, field_length, places, output):
    """ Write right justified double - user specified decimal places """
    
    data_text = fix_decimals(data, places)
    if field_length < len(data_text):
        field_length_error(field_length, output)
        return
    
    padding = pad[0:(field_length - len(data_text))]
    output.write(padding + data_text)
    

def center_places(data, field_length, places, output):
    """ Write centered double - user specified decimal places """
    
    data_text = fix_decimals(data, places)
    if field_length < len(data_text):
        field_length_error(field_length, output)
        return
    
    total_pad_length = field_length - len(data_text)
    end_pad_length = int(total_pad_length / 2)
    start_pad_length = total_pad_length - end_pad_length
    
    start_padding = pad[0:start_pad_length]
    end_padding = pad[0:end_pad_length]
    output.write(start_padding + data_text + end_padding)
    
    
""" AUXILIARY METHODS """

def field_length_error(field_length, output):
    for i in range(1, field_length+1):
        output.write("#")
        
        
def fix_decimals(data, places):
    data = int(data * math.pow(10, places)) / math.pow(10, places)
    return str(data)