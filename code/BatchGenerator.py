import numpy as np
import random
import copy
import os
DEMOPATH = "../dataset/Demo4FED/"
class BatchConfig():
    def __init__(self,batchsize,timestep,randompathcount):
        self.batchsize = batchsize
        self.timestep = timestep
        self.randompathcount = randompathcount

class BatchGenerator():

    def __init__(self,config):
        self.token_size = 0
        self.all_path = []
        self.batchstart = 0
        self.read_data()
        self.detectPath(config.timestep, config.randompathcount)
        pass

    def generateData(self, batchsize, timestep, vecsize):
        fw_data = []
        bw_data= []
        end = self.batchstart + batchsize
        if end > len(self.all_path):
            this_batch = self.all_path[self.batchstart:] + (self.all_path[0:end-len(self.all_path)])
            self.batchstart = end-len(self.all_path)
            print(len(this_batch))
        else:
            this_batch = self.all_path[self.batchstart : end]
            self.batchstart = self.batchstart + batchsize
        X_fw = []
        Y_fw = []
        X_bw = []
        Y_bw = []
        for bno in range(batchsize):
            minibatch = []
            for ts in range(timestep):
                if ts == 0:
                    minibatch.append((self.id_2_feature['0']))
                else:
                    minibatch.append((self.id_2_feature[this_batch[bno][ts-1]]))
            X_fw.append(minibatch)

        for bno in range(batchsize):
            te = this_batch[bno].copy()
            te.append('1')
            Y_fw.append(np.array(te).astype(np.int).tolist())

        for bno in range(batchsize):
            minibatch = []
            te = this_batch[bno].copy()
            te.append('1')
            te.reverse()
            for ts in range(timestep):
                minibatch.append((self.id_2_feature[te[ts]]))
            X_bw.append(minibatch)

        for bno in range(batchsize):
            te = this_batch[bno].copy()
            te.reverse()
            te.append('0')
            Y_bw.append(np.array(te).astype(np.int).tolist())

        fw_data.append(np.array(X_fw))
        fw_data.append(np.array(Y_fw))
        bw_data.append(np.array(X_bw))
        bw_data.append(np.array(Y_bw))
        fw_data = fw_data
        bw_data = bw_data

        return fw_data,bw_data

    def generateDataStruct(self, batchsize, timestep, vecsize):
        fw_data = []
        bw_data= []

        end = self.batchstart + batchsize
        if end > len(self.all_path):
            this_batch = self.all_path[self.batchstart:] + (self.all_path[0:end-len(self.all_path)])
            self.batchstart = end-len(self.all_path)
            print(len(this_batch))
        else:
            this_batch = self.all_path[self.batchstart : end]
            self.batchstart = self.batchstart + batchsize
        X_fw = []
        Y_fw = []
        X_bw = []
        Y_bw = []
        for bno in range(batchsize):
            minibatch = []
            for ts in range(timestep):
                if ts == 0:
                    minibatch.append(0)
                else:
                    minibatch.append(int(this_batch[bno][ts-1]))
            X_fw.append(minibatch)

        for bno in range(batchsize):
            te = this_batch[bno].copy()
            te.append('1')
            Y_fw.append(np.array(te).astype(np.int).tolist())

        for bno in range(batchsize):
            minibatch = []
            te = this_batch[bno].copy()
            te.append('1')
            te.reverse()
            for ts in range(timestep):
                minibatch.append(int(te[ts]))
            X_bw.append(minibatch)

        for bno in range(batchsize):
            te = this_batch[bno].copy()
            te.reverse()
            te.append('0')
            Y_bw.append(np.array(te).astype(np.int).tolist())

        fw_data.append(np.array(X_fw))
        fw_data.append(np.array(Y_fw))
        bw_data.append(np.array(X_bw))
        bw_data.append(np.array(Y_bw))
        fw_data = fw_data
        bw_data = bw_data

        return fw_data,bw_data

    def detectPath(self, path_length, sample_size):
        stack = []
        sta = 0
        for key in self.origin_graph.keys():
            stack.append(key)
            visited = set()
            sample_count = 0
            while len(stack) > 0 :
                top = stack[-1]
                if top in self.origin_graph.keys() and len(stack) < path_length:
                    preobj = self.origin_graph[top]
                    v_c = 0
                    for asa in preobj:
                        if asa in visited:
                            v_c = v_c + 1
                    i = random.randint(0,len(preobj)-1)
                    while preobj[i] in visited and v_c < len(preobj):
                        i = random.randint(0, len(preobj) - 1)
                    if v_c == len(preobj):
                        if len(stack) == 1:
                                break
                        stack.pop()
                        stack.pop()
                    else:
                        value = preobj[i]
                        visited.add(value)
                        pre = value.split("\t\t")[0]
                        val = value.split("\t\t")[-1]
                        if val in stack:
                            if stack.index(val) % 2 == 1:
                                stack.append(pre)
                                stack.append(val)
                        else:
                            stack.append(pre)
                            stack.append(val)
                elif len(stack) == path_length:
                    te = []
                    for node in stack:
                        name = self.node_id[node]
                        te.append(name)
                    self.all_path.append(te)
                    sample_count = sample_count + 1
                    if sample_count >= sample_size:
                        break
                    if len(stack) == 1:
                        break
                    stack.pop()
                    stack.pop()
                else:
                    if len(stack) == 1:
                        break
                    stack.pop()
                    stack.pop()
            visited.clear()
            stack.clear()
            sta = sta+1
            if sta % 1000 == 0:
                print(str(len(self.all_path)) + " : " + str(sta/1000))
        random.shuffle(self.all_path)
        self.all_train_token = len(self.all_path)

    def read_data(self):

        read_feature = open(os.path.join(DEMOPATH,"Graph_Node_Feature_Handled.txt"), 'r', encoding='utf-8')
        read_origin_node_id = open(os.path.join(DEMOPATH,"Graph_Origin_Node_ID.txt"), 'r', encoding='utf-8')
        read_graph = open(os.path.join(DEMOPATH,"Graph_Uri.txt"), 'r', encoding='utf-8')
        self.id_2_feature = {}
        self.origin_graph = {}
        self.node_id = {}
        for line in read_feature:
            res = line.split("\t\t")[1]
            res = res.split(' ')[0:-1]
            self.id_2_feature[line.split("\t\t")[0]] = np.array(res).astype(np.float32).tolist()
        read_feature.close()
        temp_str = ""
        for line in read_origin_node_id:
            line = line.replace('\n', '')
            try:
                if len(line.split("\t\t")) == 2 and len(temp_str)==0:
                    self.node_id[line.split("\t\t")[0]] = line.split("\t\t")[-1]
                elif len(line.split("\t\t")) == 2 and len(temp_str)>0:
                    self.node_id[temp_str+line.split("\t\t")[0]] = line.split("\t\t")[-1]
                    temp_str=""
                else:
                    temp_str = temp_str+line
            except IndexError:
                print(line)
        read_origin_node_id.close()

        temp_str = ""
        temp_str_pre = ""
        temp_key = ""
        all_uri = 0
        for line in read_graph:
            line = line.replace('\n', '')
            res = line.split("\t\t")
            all_uri = all_uri+1
            if len(res) == 3 and len(temp_str) == 0:
                if res[0] in self.origin_graph.keys():
                    self.origin_graph[res[0]].append(res[1] + "\t\t" + res[2])
                else:
                    self.origin_graph[res[0]] = [res[1] + "\t\t" + res[2]]
                temp_key = res[0]
            elif len(res) == 3 and len(temp_str) > 0:
                if res[0] in self.origin_graph.keys():
                    self.origin_graph[res[0]].append(res[1] + "\t\t" + res[2])
                else:
                    self.origin_graph[temp_str+res[0]] = [res[1] + "\t\t" + res[2]]
                temp_str = ""
                temp_key = res[0]
            temp_str_pre = line


        print(len(self.id_2_feature))
        print(len(self.origin_graph))
        print(len(self.node_id))
        print(all_uri-2)
        self.token_size = len(self.id_2_feature)




class FineTuningBatchGenerator():
    def __init__(self):
        self.batchStart = 0
        self.sampleUniSize = 0
        self.initialize_original_data()
        pass

    def initialize_original_data(self):
        read_feature = open("../DBpedia_Data/Graph_Node_Feature_Handled.txt", 'r', encoding='utf-8')
        read_origin_node_id = open("../DBpedia_Data/Graph_Origin_Node_ID.txt", 'r', encoding='utf-8')
        read_graph = open("../DBpedia_Data/Original_Graph_Uri.txt", 'r', encoding='utf-8')



        self.id_2_feature = {}
        self.origin_graph = {}
        self.node_id = {}
        self.training_ids = []
        self.all_training_instance = []
        self.all_labels = []
        for line in read_feature:
            res = line.split("\t\t")[1]
            res = res.split(' ')[0:-1]
            self.id_2_feature[line.split("\t\t")[0]] = np.array(res).astype(np.float32).tolist()
        read_feature.close()
        temp_str = ""
        for line in read_origin_node_id:
            line = line.replace('\n', '')
            try:
                if len(line.split("\t\t")) == 2 and len(temp_str)==0:
                    self.node_id[line.split("\t\t")[0]] = line.split("\t\t")[-1]
                elif len(line.split("\t\t")) == 2 and len(temp_str)>0:
                    self.node_id[temp_str+line.split("\t\t")[0]] = line.split("\t\t")[-1]
                    temp_str=""
                else:
                    temp_str = temp_str+line
            except IndexError:
                print(line)
        read_origin_node_id.close()

        all_uri = 0
        for line in read_graph:
            line = line.replace('\n', '')
            res = line.split("\t\t")
            all_uri = all_uri + 1
            if len(res) == 4:
                if res[0] not in self.origin_graph.keys():
                    self.origin_graph[res[0]] = [line]
                else:
                    val = self.origin_graph[res[0]]
                    val.append(line)
        read_graph.close()

        for key in self.node_id:
            if key in self.origin_graph.keys():
                val = self.origin_graph[key]
                one_count = 0
                for li in val:
                    if li.split("\t\t")[-1] == '1':
                        one_count += 1

                if len(val) > 20 and len(val) < 100 and one_count > 0 and one_count < len(val):
                    self.training_ids.append(key)

        for entity in self.training_ids:
            triples = self.origin_graph[entity]
            for triple in triples:
                tri = triple.split("\t\t")
                subject = self.id_2_feature[self.node_id[tri[0]]]
                predicate = self.id_2_feature[self.node_id[tri[1]]]
                object = self.id_2_feature[self.node_id[tri[2]]]
                label = int(tri[-1])
                self.all_training_instance.append([subject,predicate,object])
                self.all_labels.append(label)


        self.all_instance_size = len(self.all_training_instance)
        print(len(self.id_2_feature))
        print(len(self.origin_graph))
        print(len(self.node_id))
        print(len(self.training_ids))
        print(self.all_instance_size)
        print(all_uri-2)
        self.token_size = len(self.id_2_feature)

    def sample_from_triple_set(self,key,triples,topkSize, sampleSize): #topkSize * 3 * vecsize ,sampleSize = 1
        instance = []
        label = []
        choose = []
        while len(choose) < sampleSize:
            sampleSum = set()
            while len(sampleSum) < topkSize:
                chooseId = random.randint(0,len(triples)-1)
                sampleSum.add(chooseId)
            if sampleSum not in choose:
                choose.append(sampleSum)

        for cSum in choose:
            tokens_fw = []
            tokens_bw = []
            tokenLabel_fw = []
            tokenLabel_bw = []
            for id in cSum:
                tri = triples[id]
                vals = tri.split("\t\t")
                keyFeature = self.id_2_feature[self.node_id[vals[0]]]
                propFeature = self.id_2_feature[self.node_id[vals[1]]]
                valFeature = self.id_2_feature[self.node_id[vals[2]]]
                # ins = keyFeature + propFeature + valFeature
                tokens_fw.append([keyFeature,propFeature,valFeature])
                tokens_bw.append([valFeature,propFeature,keyFeature])
                tokenLabel_fw.append(int(vals[-1]))
                tokenLabel_bw.append(int(vals[-1]))
            instance.append(np.array(tokens_fw))
            instance.append(np.array(tokens_bw))
            label.append(np.array([np.array(tokenLabel_fw)]).transpose())
            label.append(np.array([np.array(tokenLabel_bw)]).transpose())
        return instance,label

    def sample_from_triple_set_kge(self,key,triples,topkSize, sampleSize,name): #topkSize * 3 * vecsize ,sampleSize = 1
        instance = []
        label = []
        choose = []
        while len(choose) < sampleSize:
            sampleSum = set()
            while len(sampleSum) < topkSize:
                chooseId = random.randint(0,len(triples)-1)
                sampleSum.add(chooseId)
            if sampleSum not in choose:
                choose.append(sampleSum)

        for cSum in choose:
            tokens_fw = []
            tokens_bw = []
            tokenLabel_fw = []
            tokenLabel_bw = []
            for id in cSum:
                tri = triples[id]
                vals = tri.split("\t\t")
                feature = []
                # ins = keyFeature + propFeature + valFeature
                tokens_fw.append(feature)
                tokens_bw.append(feature)
                tokenLabel_fw.append(int(vals[-1]))
                tokenLabel_bw.append(int(vals[-1]))
            instance.append(np.array(tokens_fw))
            instance.append(np.array(tokens_bw))
            label.append(np.array([np.array(tokenLabel_fw)]).transpose())
            label.append(np.array([np.array(tokenLabel_bw)]).transpose())
        return instance,label

    def generate_fine_tuning_data(self,batchSize,topkSize,sampleSize):
        start = self.batchStart
        key = self.training_ids[start]
        triples = self.origin_graph[key]
        if self.sampleUniSize == 0:
            self.batchStart = (self.batchStart + 1) % len(self.training_ids)
            start = self.batchStart
            key = self.training_ids[start]
            triples = self.origin_graph[key]
            self.sampleUniSize = int(len(triples) / 10) * sampleSize
        instance,label = self.sample_from_triple_set(key,triples,topkSize,1)
        self.sampleUniSize -= 1
        return instance,label


    def generate_fine_tuning_data_by_batch(self,batchSize):
        tokens_fw = []
        tokens_bw = []
        tokenLabel_fw = []
        tokenLabel_bw = []

        end = self.batchStart + batchSize
        if end > self.all_instance_size:
            this_batch = self.all_training_instance[self.batchStart:] + (self.all_training_instance[0:end - self.all_instance_size])
            labels =  self.all_labels[self.batchStart:] + (self.all_labels[0:end - self.all_instance_size])
            self.batchStart = end - self.all_instance_size
        else:
            this_batch = self.all_training_instance[self.batchStart: end]
            labels = self.all_labels[self.batchStart: end]
            self.batchStart = self.batchStart + batchSize

        tokens_bw = [copy.deepcopy(i) for i in this_batch]
        [i.reverse() for i in tokens_bw]
        tokens_bw = np.array(tokens_bw)
        tokens_fw = np.array(this_batch)
        tokenLabel_fw = np.array([np.array(labels)]).transpose()
        tokenLabel_bw = tokenLabel_fw
        return [tokens_fw,tokens_bw], [tokenLabel_fw,tokenLabel_bw]


    def generate_fine_tuning_data_Kge(self,batchSize,name,sampleSize,topkSize):
        start = self.batchStart
        key = self.training_ids[start]
        triples = self.origin_graph[key]
        if self.sampleUniSize == 0:
            self.batchStart = (self.batchStart + 1) % len(self.training_ids)
            start = self.batchStart
            key = self.training_ids[start]
            triples = self.origin_graph[key]
            self.sampleUniSize = int(len(triples) / 10) * sampleSize
        instance,label = self.sample_from_triple_set_kge(key,triples,topkSize,1,name)
        self.sampleUniSize -= 1
        return instance,label

if __name__ == '__main__':
    batchconfig = BatchConfig(batchsize=256, timestep=7, randompathcount=100)
    BG = BatchGenerator(batchconfig)




















