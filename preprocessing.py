import csv

readfile = open('data.csv', 'r')
readCSV = csv.reader(readfile, delimiter=',')

writefile = open('raj.txt','w')
for row in readCSV:
    row = ''.join(row).replace("N;","").replace('""N','').replace('"N','')
    #writefile.write(row)
    row = row.split(";")
    for line in row:
        if len(line)>1:
            line = line.replace('"','')
            writefile.write(line+ '\n')
    '''for str in row:
        if len(str)>1:
            
              '''  
writefile.close()
readfile.close()
writefile = open('processed.txt','w')
uniqlines = set(open('raj.txt').readlines())
for line in uniqlines:
    writefile.write(line)
writefile.close()
    