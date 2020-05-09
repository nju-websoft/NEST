import sys
import os
dataSetRootPath = "../dataset/Benchmarks/Output"
def doBenchMark4FED(file,k):
    ent = {}
    with open(file,encoding='utf-8') as f:
        for entity in f:
            entity = entity.strip()
            triple = entity.split("\t\t")[0]
            score = entity.split("\t\t")[-1]
            head = triple.split("\t")[0]
            if head not in ent.keys():
                ent[head] = [score]
            else:
                ent[head].append(score)
    F1 = 0
    for head in ent:
        precision = 0.0
        recall = 0.0
        for score in ent[head]:
            if k == 5:
                precision += (float)(score.split(" ")[0])
                recall += (float)(score.split(" ")[0])
            else:
                precision += (float)(score.split(" ")[-1])
                recall += (float)(score.split(" ")[-1])
        precision /= (8*k)
        recall /= (8*k)
        F1 += 2/(1.0/precision + 1.0/recall)
    F1 /= len(ent)
    return F1

def doBenchMark4ESBM(file,k):
    scores = []
    with open(file,encoding='utf-8') as f:
        for entity in f:
            entity = entity.strip()
            score = entity.split("\t\t")[-1]
            scores.append(score)
    topk_true = 0.0
    for score in scores:
        score = score.split(" ")
        tk = score[int((k / 5 - 1) * 6):int((k / 5) * 6)]
        for s in tk:
            topk_true += float(s)
    return topk_true/(6 * len(scores))



if __name__ == "__main__":
    if len(sys.argv) > 1:
        datasetName = sys.argv[1]
    else:
        datasetName = ""

    ESBM_D_Top5 = doBenchMark4ESBM(os.path.join(dataSetRootPath,"ESBMDTop5"), 5)
    ESBM_D_Top10 = doBenchMark4ESBM(os.path.join(dataSetRootPath,"ESBMDTop10"), 10)
    ESBM_L_Top5 = doBenchMark4ESBM(os.path.join(dataSetRootPath,"ESBMLTop5"), 5)
    ESBM_L_Top10 = doBenchMark4ESBM(os.path.join(dataSetRootPath,"ESBMLTop10"), 10)
    FED_Top5 = doBenchMark4FED(os.path.join(dataSetRootPath,"FEDTop5"), 5)
    FED_Top10 = doBenchMark4FED(os.path.join(dataSetRootPath,"FEDTop10"), 10)
    if datasetName == "ESBM":
        print("ESBM-D Top5 F1 : " + str(ESBM_D_Top5))
        print("ESBM-D Top10 F1 : " + str(ESBM_D_Top10))
        print("ESBM-L Top5 F1 : " + str(ESBM_L_Top5))
        print("ESBM-L Top10 F1 : " + str(ESBM_L_Top10))
    elif datasetName == "FED":
        print("FED Top5 F1 : " + str(FED_Top5))
        print("FED Top10 F1 : " + str(FED_Top10))
    else:
        print("ESBM-D Top5 F1 : " + str(ESBM_D_Top5))
        print("ESBM-D Top10 F1 : " + str(ESBM_D_Top10))
        print("ESBM-L Top5 F1 : " + str(ESBM_L_Top5))
        print("ESBM-L Top10 F1 : " + str(ESBM_L_Top10))
        print("FED Top5 F1 : " + str(FED_Top5))
        print("FED Top10 F1 : " + str(FED_Top10))