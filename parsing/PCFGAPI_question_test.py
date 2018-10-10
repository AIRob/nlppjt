#coding=utf-8
#stanford parser  test

from PCFGAPI import StanfordPCFGAPI

def main():
    string = '高血压与高血脂头晕眼花工作又忙应该怎么防治？'
    sfpcfgapi = StanfordPCFGAPI(string)
    seg_str = sfpcfgapi.get_seg_str()
    sfpcfgapi.stanford_ch_parse(seg_str)

if __name__ == '__main__':
    main()
    