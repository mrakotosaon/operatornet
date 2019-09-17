'''
Created on May 6, 2017

@author: optas
'''


def trim_content_after_last_dot(s):
    '''Example: if s = myfile.jpg.png, returns myfile.jpg
    '''
    index = s[::-1].find('.') + 1
    s = s[:len(s) - index]
    return s
