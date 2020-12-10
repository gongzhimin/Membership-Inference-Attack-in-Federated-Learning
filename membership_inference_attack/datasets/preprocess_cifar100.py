import pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

traindict = unpickle("./cifar-100-python/train")
testdict  = unpickle("./cifar-100-python/test")

with open("cifar100.txt", "w+") as f:
    for i in range(len(traindict[b'data'])):
        a = ','.join([str(c) for c in traindict[b'data'][i][:1024]]) + ';' + \
            ','.join([str(c) for c in traindict[b'data'][i][1024:2048]]) + ';' + \
            ','.join([str(c) for c in traindict[b'data'][i][2048:]]) + ';' + \
            str(traindict[b'fine_labels'][i])
        f.write(a+"\n")

with open("cifar100.txt", "a") as f:
    for i in range(len(testdict[b'data'])):
        a = ','.join([str(c) for c in testdict[b'data'][i][:1024]]) + ';' + \
            ','.join([str(c) for c in testdict[b'data'][i][1024:2048]]) + ';' + \
            ','.join([str(c) for c in testdict[b'data'][i][2048:]]) + ';' + \
            str(testdict[b'fine_labels'][i])
        f.write(a+"\n")