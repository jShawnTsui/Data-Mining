import numpy as np 
import re
import gzip
from itertools import islice

def readData(file_name):
    length = 0
    data_list = list()
    with gzip.open(file_name, 'rt') as file:
        while True:
            lines = list(islice(file, 100))
            if not lines:
                break

            users = re.findall(r"user(?=([0-9 .:]+))", ''.join(lines))
            length  += len(users)
            users_str = ''.join(users)
            fearures = [feature.split(':')[1] for feature in users_str.split()]
            data_list.extend(fearures)
    data = np.array(data_list).astype(np.float)
    data.resize(length, 6)
    # np.save('data', data)
    return data

def main():
    file_name = './R6/ydata-fp-td-clicks-v1_0.20090501.gz'
    readData(file_name)

if __name__ == '__main__':
    main()

