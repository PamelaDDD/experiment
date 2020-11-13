import jieba
def main():
    w = open('./data/eng-ch.txt','w',encoding='utf-8')
    with open("./data/cmn.txt","r",encoding='utf-8') as f:
        for line in f.readlines():
            line = line.split('\t')
            eng,ch = line[0],line[1]
            newline = "{}\t{}\n".format(eng,ch)
            w.writelines(newline)
    w.close()


# if __name__ == '__main__':
#     main()


words = jieba.lcut("我觉得当汤姆发现他买来的画是赝品的时候，他会很生气。")
print(type(words))
print(" ".join(words))