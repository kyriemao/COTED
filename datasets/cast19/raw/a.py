from IPython import embed
with open('2019qrels.txt', 'r') as f:
    data = f.readlines()


d = set()
for line in data:
    line = line.strip().split(' ')
    t = line[0].split('_')[0]
    if line[3] != '0':
        d.add(t)

print(len(d))