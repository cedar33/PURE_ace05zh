from xml.etree import ElementTree
from bs4 import BeautifulSoup
import nltk
import json
import re
import argparse
import os
from tqdm import tqdm

class Parser:
    def __init__(self, path):
        self.entity_mentions = []
        self.event_mentions = []
        self.sentences = []
        self.doc_key = path.split("/")[-1]
        self.entity_mentions, self.event_mentions = self.parse_xml(path + '.apf.xml')
        self.sents_with_pos = self.parse_sgm(path + '.sgm')
        self.prefix_length = self.get_prefix_len(path + '.sgm')

    @staticmethod
    def clean_text(text):
        text = text.replace('\n', ' ')
        return text

    def get_prefix_len(self, path):
        xml_str = "".join(line for line in open(path, "r", encoding='utf-8'))
        soup = BeautifulSoup(xml_str)
        prefix_length = 1
        ignore_key = ["docid", "doctype", "datetime", "headline"]
        for k in ignore_key:
            try:
                prefix_length += (len(soup.find(k).text)+1)
            except AttributeError:
                print(k)
        # try:
        #     prefix_length = len(soup.find("docid").text)+len(soup.find("doctype").text)+len(soup.find("datetime").text)+len(soup.find("headline").text)
        # except AttributeError as e:
        #     print(xml_str)
        #     raise e
        return prefix_length

    def get_data(self):
        sts_len = 0
        data = dict()
        data["doc_key"] = self.doc_key
        data["sentences"] = []
        data["ner"] = []
        def clean_text(text):
            text = text.replace('\n', ' ')
            return text

        for idx, sent in enumerate(self.sents_with_pos):
            if idx == 0:
                continue
            item = dict()
            item['sentence'] = clean_text(sent['text'])
            item['position'] = sent['position']
            text_position = sent['position']

            for i, s in enumerate(item['sentence']):
                if s != ' ':
                    item['position'][0] += i
                    break
            item['sentence'] = item['sentence'].strip()

            entity_map = dict()
            item['golden-entity-mentions'] = []
            item['golden-event-mentions'] = []

            for entity_mention in self.entity_mentions:
                entity_position = entity_mention['position']
                if text_position[0] <= entity_position[0] and entity_position[1] <= text_position[1]:
                    item['golden-entity-mentions'].append({
                        'text': clean_text(entity_mention['text']),
                        'phrase-type': entity_mention['phrase-type'],
                        'position': entity_position,
                        'entity-type': entity_mention['entity-type'],
                    })
                    entity_map[entity_mention['entity-id']] = entity_mention

            for event_mention in self.event_mentions:
                event_position = event_mention['trigger']['position']
                if text_position[0] <= event_position[0] and event_position[1] <= text_position[1]:
                    event_arguments = []
                    for argument in event_mention['arguments']:
                        try:
                            entity_type = entity_map[argument['entity-id']]['entity-type']
                        except KeyError:
                            print('[Warning] The entity in the other sentence is mentioned. This argument will be ignored.')
                            continue

                        event_arguments.append({
                            'role': argument['role'],
                            'position': argument['position'],
                            'entity-type': entity_type,
                            'text': self.clean_text(argument['text']),
                        })

                    item['golden-event-mentions'].append({
                        'trigger': event_mention['trigger'],
                        'arguments': event_arguments,
                        'position': event_position,
                        'event_type': event_mention['event_type'],
                    })
            # if item["golden-entity-mentions"] or item["golden-event-mentions"]:
            if len(item["sentence"]) < 512:
                data["sentences"].append(item["sentence"])

                sentence_ner = []
                if item["golden-entity-mentions"]:
                    for d in item["golden-entity-mentions"]:
                        # print(self.prefix_length, d["position"], )
                        sentence_ner.append([d["position"][0]-item['position'][0]+sts_len, d["position"][1]-item['position'][0]+sts_len, d["entity-type"].split(":")[0]])
                if item["golden-event-mentions"]:
                    for d in item["golden-event-mentions"]:
                        sentence_ner.append([d["position"][0]-item['position'][0]+sts_len, d["position"][1]-item['position'][0]+sts_len, d["event_type"].split(":")[0]])
                        for _d in d["arguments"]:
                            if "entity-type" in _d:
                                sentence_ner.append([_d["position"][0]-item['position'][0]+sts_len, _d["position"][1]-item['position'][0]+sts_len, _d["entity-type"].split(":")[0]])
                data["ner"].append(sentence_ner)
                sts_len += len(item["sentence"])
            else:
                self.prefix_length += len(item["sentence"])
        assert len(data["sentences"]) == len(data["ner"]), (len(data["ner"]), len(data["sentences"]))
        return data

    @staticmethod
    def parse_sgm(sgm_path):
        with open(sgm_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), features='html.parser')
            sgm_text = soup.text

            doc_type = soup.doc.doctype.text.strip()

            def remove_tags(selector):
                tags = soup.findAll(selector)
                for tag in tags:
                    tag.extract()

            if doc_type == 'WEB TEXT':
                remove_tags('poster')
                remove_tags('postdate')
                remove_tags('subject')
            elif doc_type in ['CONVERSATION', 'STORY']:
                remove_tags('speaker')

            converted_text = soup.text
            # converted_text = converted_text.replace(' ill. ', ' ill ')
            # for sent in nltk.sent_tokenize(converted_text):
            #     sents.extend(re.split('[\n\n．]', sent))
            sents = re.split('。|！|\!|\n\n|？|\?', converted_text)
            sents = list(filter(lambda x: len(x) > 5, sents))
            sents = sents[1:]
            sents_with_pos = []
            last_pos = 0
            texts = [word for word in sgm_text]
            for sent in sents:
                sent =sent.strip()
                pos = sgm_text.find(sent, last_pos)
                last_pos = pos
                sents_with_pos.append({
                    'text': sent,
                    'position': [pos, pos + len(sent)]
                })

            return sents_with_pos

    def parse_xml(self, xml_path):
        entity_mentions, event_mentions = [], []
        tree = ElementTree.parse(xml_path)
        root = tree.getroot()

        for child in root[0]:
            if child.tag == 'entity':
                entity_mentions.extend(self.parse_entity_tag(child))
            elif child.tag in ['value', 'timex2']:
                entity_mentions.extend(self.parse_value_timex_tag(child))
            elif child.tag == 'event':
                event_mentions.extend(self.parse_event_tag(child))

        return entity_mentions, event_mentions

    @staticmethod
    def parse_entity_tag(node):
        entity_mentions = []

        for child in node:
            if child.tag != 'entity_mention':
                continue
            extent = child[0]
            charset = extent[0]

            entity_mention = dict()
            entity_mention['entity-id'] = child.attrib['ID']
            entity_mention['phrase-type'] = child.attrib['TYPE']
            entity_mention['entity-type'] = '{}:{}'.format(node.attrib['TYPE'], node.attrib['SUBTYPE'])
            entity_mention['text'] = charset.text
            entity_mention['position'] = [int(charset.attrib['START']), int(charset.attrib['END'])]

            entity_mentions.append(entity_mention)

        return entity_mentions

    @staticmethod
    def parse_event_tag(node):
        event_mentions = []
        for child in node:
            if child.tag == 'event_mention':
                event_mention = dict()
                event_mention['event_type'] = '{}:{}'.format(node.attrib['TYPE'], node.attrib['SUBTYPE'])
                event_mention['arguments'] = []
                for child2 in child:
                    if child2.tag == 'ldc_scope':
                        charset = child2[0]
                        event_mention['text'] = charset.text.replace('\n', ' ')
                        event_mention['position'] = [int(charset.attrib['START']), int(charset.attrib['END'])]
                    if child2.tag == 'anchor':
                        charset = child2[0]
                        event_mention['trigger'] = {
                            'text': charset.text.replace('\n', ' '),
                            'position': [int(charset.attrib['START']), int(charset.attrib['END'])],
                        }
                    if child2.tag == 'event_mention_argument':
                        extent = child2[0]
                        charset = extent[0]
                        if 'Time-' in child2.attrib['ROLE']:
                            Role = 'Time'
                        else:
                            Role = child2.attrib['ROLE']
                        event_mention['arguments'].append({
                            'text': charset.text,
                            'position': [int(charset.attrib['START']), int(charset.attrib['END'])],
                            'role': Role,
                            'entity-id': child2.attrib['REFID']
                        })
                event_mentions.append(event_mention)
        return event_mentions

    @staticmethod
    def parse_value_timex_tag(node):
        entity_mentions = []

        for child in node:
            extent = child[0]
            charset = extent[0]

            entity_mention = dict()
            entity_mention['entity-id'] = child.attrib['ID']

            if 'TYPE' in node.attrib:
                entity_mention['entity-type'] = node.attrib['TYPE']
                entity_mention['phrase-type'] = 'NUM'
            if 'SUBTYPE' in node.attrib:
                entity_mention['entity-type'] += ':{}'.format(node.attrib['SUBTYPE'])
            if child.tag == 'timex2_mention':
                entity_mention['entity-type'] = 'TIM:time'
                entity_mention['phrase-type'] = 'TIM'

            entity_mention['text'] = charset.text
            entity_mention['position'] = [int(charset.attrib['START']), int(charset.attrib['END'])]

            entity_mentions.append(entity_mention)

        return entity_mentions

def get_data_paths(ace2005_path):
    test_files, dev_files, train_files = [], [], []

    with open('./data_list.csv', mode='r') as csv_file:
        rows = csv_file.readlines()
        for row in rows[1:]:
            items = row.replace('\n', '').split(',')
            data_type = items[0]
            name = items[1]

            path = os.path.join(ace2005_path, name)
            if data_type == 'test':
                test_files.append(path)
            elif data_type == 'dev':
                dev_files.append(path)
            elif data_type == 'train':
                train_files.append(path)
    return test_files, dev_files, train_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help="Path of ACE2005 English data", default='./data/ace_2005_td_v7/data/Chinese')
    args = parser.parse_args()
    test_files, dev_files, train_files = get_data_paths(args.data)
    f_train_json = open("./train.json", "w", encoding="utf-8")
    f_test_json = open("./test.json", "w", encoding="utf-8")
    f_dev_json = open("./dev.json", "w", encoding="utf-8")
    for f in tqdm(train_files):
        data = Parser(f).get_data()
        f_train_json.write(json.dumps(data, ensure_ascii=False)+"\n")
    for f in tqdm(dev_files):
        data = Parser(f).get_data()
        f_dev_json.write(json.dumps(data, ensure_ascii=False) +"\n")
    for f in tqdm(test_files):
        data = Parser(f).get_data()
        f_test_json.write(json.dumps(data, ensure_ascii=False) +"\n")
    f_train_json.close()
    f_test_json.close()
    f_dev_json.close()

    # data = Parser('E:\\ace_2005_td_v7_LDC2006T06\\ace_2005_td_v7\data\Chinese\\bn\\adj\CTS20001121.1300.0182').get_data()
    # with open('output/sample.json', 'w', encoding='utf-8') as f:
    #     json.dump(data, f, indent=2, ensure_ascii=False)