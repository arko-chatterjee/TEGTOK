import json

def splitDatasetFor(fileName):
    print("LOADING REDDIT DATA")
    file = open(fileName+'Augmented.json')     # SELECT FILE
    redditData = json.load(file)
    train = []
    test = []
    valid = []
    i = 2
    j = 1
    for item in redditData:
        if (i % 5 == 0):
            item['dialog_id'] = "valid-"+str(int(i/5))
            valid.append(item)
        elif (i % 5 == 1):
            item['dialog_id'] = "test-"+str(int(i/5))
            test.append(item)
        else:
            item['dialog_id'] = "train-"+str(j)
            j += 1
            train.append(item)
        i += 1
    
    print("WRITING JSON")
    json_object = json.dumps(train, indent=4)
    with open('../data/'+fileName+'/train.json', "w") as outfile:
        outfile.write(json_object)
    json_object = json.dumps(test, indent=4)
    with open('../data/'+fileName+'/test.json', "w") as outfile:
        outfile.write(json_object)
    json_object = json.dumps(valid, indent=4)
    with open('../data/'+fileName+'/valid.json', "w") as outfile:
        outfile.write(json_object)




if (__name__ == '__main__'):
    splitDatasetFor('RoastMe')
    splitDatasetFor('FreeCompliments')