from re import sub
from sklearn.cluster import KMeans
import numpy as np
# f = open('THM_001.txt')
# lines = f.readlines()
# for x in range(len(lines)):
#     lines[x] = lines[x][:-1]

# print(type(lines))
# print(lines[0])
# print(lines)

def ex_array(filename):
    """_summary_
    Args:
        filename (string): name of file
    Returns:
        list: [[w,h], ...]
    """
    f = open(filename)
    lines = f.readlines()
    res = []
    for x in range(len(lines)):
        lines[x] = lines[x][:-1]
        subarray = lines[x].split()
        subarray = subarray[3:5]
        res += [subarray]
    return res

# def ex_array(strarray):
#     res = [[]]
#     for i in range(len(strarray)):
#         subarr = substr.split()

def main():
    wh_array = []
    
    for i in range(1, 302):
        zero_filled_number = str(i).zfill(3)
        file = 'THM_' + zero_filled_number + '.txt'
        wh_array += ex_array(file)

    # print(wh_array)
    
    X = np.array(wh_array)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print('labels:')
    print(kmeans.labels_)
    
    print(kmeans.cluster_centers_)
    # table_width     = {0:[], 1:[], 2:[]}
    # table_height    = {0:[], 1:[], 2:[]}
    
    
    # for i in range(len(X)):
    #     tmp_label = kmeans.labels_[i]
    #     table_width[tmp_label] += [X[i][0]]
    #     table_height[tmp_label] += [X[i][1]]
        
            
        

if __name__ == "__main__":
    main()
