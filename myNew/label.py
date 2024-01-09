import csv
import json
# ------------------CPED数据集中speaker标签转换为pkl文件-----------
with open("../data/CPED/speakers.txt", "r", encoding='utf-8') as f:
    with open("../data/myCPED/speaker_vocab.pkl", "w") as g:
        reader = csv.reader(f)
        dict_all = {}
        dict_row = {}
        list_row = []
        count = 0
        for row in reader:
            dict_row[row[0]] = count
            list_row.append(row[0])
            count += 1
        dict_all['stoi'] = dict_row
        dict_all['itos'] = list_row
        g.write(json.dumps(dict_all, ensure_ascii=False))
        f.close()
        g.close()

# -------------------------------label标签转换为pkl文件-----------
with open("../data/myCPED/label_vocab.pkl","w") as f:
    reader = csv.reader(f)
    dict_all = {}
    dict_row = {}
    list_row = []
    for i in range(0,32):
        # print(i)
        label = bin(i)[2:].zfill(5)
        list_row.append(label)
        dict_row[i] = label
        print(label)
    list_row.append('UNK')
    dict_row[32] = 'UNK'
    dict_all['stoi'] = dict_row
    dict_all['itos'] = list_row
    f.write(json.dumps(dict_all, ensure_ascii=False))
    f.close()


