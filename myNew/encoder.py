import csv
import json

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, AutoModelForMaskedLM, BertTokenizer, \
    BertModel

# --------------------------加载模型---------------------------------------
# config = AutoConfig.from_pretrained("../../bert/config.json")
# model = AutoModelForMaskedLM.from_pretrained("../../bert/pytorch_model.bin", config=config)
# tokenizer = AutoTokenizer.from_pretrained("../../bert/",
#                                           config="../../bert/tokenizer_config.json",
#                                           vocab_file="../../bert/vocab.txt")
config = AutoConfig.from_pretrained("../../bert/config.json")
tokenizer = BertTokenizer.from_pretrained("../../bert/",
                                          config="../../bert/tokenizer_config.json",
                                          vocab_file="../../bert/vocab.txt")
model = BertModel.from_pretrained("../../bert/pytorch_model.bin", config=config)

# -------------------------打开myCPED.csv文件------------------------------
with open("../data/myCPED/tarin_data_bert.json.feature", "w", encoding="utf-8") as file:
    with open("../data/myCPED/myCPED.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            input_text = row[3]
            # 使用tokenizer对输入文本进行编码
            input_ids = tokenizer(input_text, return_tensors="pt", truncation=True)
            output = model(**input_ids).pooler_output.tolist()
            # summaries = model.generate(
            #     input_ids=input_ids, max_length=128
            # )
            # print(summaries)
            # print(model.encode(input_ids))
            # 输出编码后的文本
            dict = {}
            dict['text'] = row[3]
            dict['speaker'] = row[0]
            dict['feature'] = output
            dict['label'] = row[4]
            dict['ent'] = row[2]
            file.write(json.dumps(dict,ensure_ascii=False))

#
# input = tokenizer("你是一个好人",return_tensors="pt")
# o = model(**input)
# print(o.pooler_output.shape)`