#coding=utf-8
#stanford parser  test

from PCFGAPI import StanfordPCFGAPI

def main():
    string = '我老公3月22日因为血压高高压190低压140，无力，身体发虚，做血项检查发现肌酐高143，正常值上限100，尿蛋白一个其它项目正常过年这两个月睡觉很晚，饮食无规律，经常喝酒，前半个月住院治疗，静点肾康，口服肾衰宁，金水宝，硝苯地平，依娜普利，盖三醇，血压降到高压130多，低压100左右，复检血肌肝没降尿蛋白正常，请问医生现在应该怎样治疗，谢谢'
    sfpcfgapi = StanfordPCFGAPI(string)
    seg_str = sfpcfgapi.get_seg_str()
    sfpcfgapi.stanford_ch_parse(seg_str)

if __name__ == '__main__':
    main()
