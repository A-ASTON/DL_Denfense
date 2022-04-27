import os
import platform
import sys
import os.path
import shutil
import json 
import numpy as np
import pickle
import copy
from collections import Counter

import warnings
warnings.filterwarnings("ignore")


class TrivalException(Exception):
    def __init__(self, info):
        self.info = info

    def __str__(self):
        printInColor(self.info, 'black','red')

def isLinux():
    isLinux = (platform.system().lower() == 'linux')
    return isLinux
    
def correctFilePath(inpath):
    isLinux = (platform.system().lower() == 'linux')
    opath = inpath.replace('\\','/')
    while True:
        if '//' in opath:
            opath = opath.replace('//','/')
        else:
            break
                
    if isLinux == False:
        opath = inpath.replace('/','\\')
        
    return opath    
    
    
def printTable(list2d, col_labels=None):
    import prettytable as pt
    tb = pt.PrettyTable()
    
    tb.padding_width = 1 # One space between column edges and contents (default) 

    if col_labels is not None:
        tb.field_names = col_labels
        for xx in col_labels:
            tb.align[xx] = 'l'

    for row in list2d:
        tb.add_row(row)
    print(tb)
    return tb
    
def genTableFigure(list2d, col_labels=None, row_labels=None, 
        col_width=0.2, table_width=0.5, table_height=0.5):
        
    import matplotlib.pyplot as plt
   
    fig = plt.figure()
    ax = fig.gca()
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)  

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)    
    
    nCols = len(list22[0])
    tb = ax.table(
        cellText=list2d, 
		colWidths=[col_width]*nCols,
        rowLabels=row_labels, 
        colLabels=col_labels,
        #rowColours=row_colors, 
        #colColours=row_colors,
		rowLoc='left',		    # row header
		colLoc='center',		# text in col head
		cellLoc='center',		# text in each data cell
		bbox=[0,0,table_width, table_height], # col, row, width, height. percent in the axis.
		edges='closed',
        loc='center')

    return fig
    
        
def getDictValue(indict, keyAddr):
    inkeys = keyAddr.split('.')

    tmpdict = indict
    for keyname in inkeys:
        tmpkeys = tmpdict.keys()
        if keyname in tmpkeys:
            subdict = tmpdict[keyname]
            tmpdict = subdict
        else:
            return None
    return subdict    
    
def setDictValue(indict, keyAddr, value):
    inkeys = keyAddr.split('.')
    
    tmpdict = indict
    for k in range(len(inkeys)-1):
        keyname = inkeys[k]

        tmpkeys = tmpdict.keys()
        if keyname in tmpkeys:
            subdict = tmpdict[keyname]
            tmpdict = subdict
        else:
            return None
    subdict[inkeys[-1] ] = value
    return value

def printStill(info):
    sys.stdout.write(info + '\r')
    os.sys.stdout.flush()
    
def printInColor(content, back_color='black', front_color='blue'):

    colorList = ['black', 'red','green','yellow','blue',
        'purple','cyan','white']
    bstr = '4'+str(colorList.index(back_color))
    fstr = '3'+str(colorList.index(front_color))
    
    import colorama
    from colorama import init,Fore,Back,Style
    init(autoreset=True)
    
    tmpstr = '\033[1;%s;%sm''%s' % (fstr, bstr, content)
    print(tmpstr)

def addToEnvPath(inpaths):
    tmpPathList = []
    if isinstance(inpaths, str):
        tmpPathList.append(inpaths)
    elif isinstance(inpaths, list):
        tmpPathList = inpaths

    for tmppath in tmpPathList:
        myassert(os.path.exists(tmppath), tmppath+" not exist")
        os.sys.path.append(tmppath)

def myassert(flag, infostr):
  if flag == False:
    print(infostr)
    assert(False)
    
def getFileHash(filepath):
    import hashlib
    
    f = open(filepath, 'rb')
    line = f.readline()
    hash = hashlib.md5()
    while(line):
        hash.update(line)
        line = f.readline()
    f.close()
    
    return hash.hexdigest()
    
def isFileEqual(file1,file2):   
    str1 = getFileHash(file1)
    str2 = getFileHash(file2)
    return str1 == str2
  
def readAFile(file):
    buffer = []
    if not os.path.exists(file):
        print("%s not exist\n" % file)
        return buffer
        
    ptrfile = open(file,"r", encoding='utf-8')
    content = ptrfile.readlines()
    ptrfile.close()
    
    _Len = len(content)
    for i in range(_Len):
        content[i] = content[i].strip()
        if len(content[i]) > 0:
            buffer.append(content[i])
    return buffer
    
def clamp(inlist, start,end):
    if isinstance(inlist, list):
        for x in inlist:
            if x < start:
                x = start
            if x > end:
                x = end
        return inlist
    else:
        x = inlist
        if x < start: 
            x = start
        if x > end:
            x = end
        return x

def sortrows(mtx, colidx):
    tmp = mtx[:,colidx]
    idx = tmp.argsort()
    return mtx[idx]
    
# column index: >= 0
def readATable(file, IDColK, dataColKList,splitter=' '):
    con = readAFile(file)
    
    idList = []    
    mtx = np.zeros([0,len(dataColKList)])
    for line in con:
        xlst = line.split(splitter)
        id = xlst[IDColK]
        idList.append(id)
        
        xlst2 = np.array(xlst)[dataColKList].astype('float')
        mtx = np.append(mtx,[xlst2],axis=0)
    return idList,mtx
            
def loadJson(json_file):
    with open(json_file,'r', encoding='utf-8') as f:
        json_out = json.loads(f.read())
    return json_out

def writeJson(ofile, dictData):  
    tmpstr = json.dumps(dictData, ensure_ascii=False,indent=4)
    fid = open(ofile, 'w', encoding='utf-8')
    fid.write(tmpstr)
    fid.close()

def charCountInStr(str,char):    
    charCount = {}
    for ch in str:
        charCount[ch] = charCount.get(ch, 0) + 1
    
    if charCount.has_key(char):
        return charCount[char]        
    else:
        return 0

def extractColumnFromFile(listfile, colIdx=0):
    nameList = []
    
    if len(listfile) > 0:
        con = readAFile(listfile)
        for line in con:
            xx = line.split()
            if len(xx) > colIdx:
                nameList.append(line.split()[colIdx])
    return nameList
    
def genFileList(inpath, fextList,fullpath=False):
    filenames = os.listdir(inpath)
    
    alist = []
    for filename in filenames:        
        fpath = os.path.join(inpath,filename)
        if not os.path.isfile(fpath):
            continue
            
        f,fext=os.path.splitext(filename)        
        tmpstr = filename
        if(fullpath):
            tmpstr = fpath
            
        if(len(fextList) > 0):
            if fext in fextList:
                alist.append(tmpstr)
        else:
            alist.append(tmpstr)
            
    return alist
    

  
def _list_dir_helper_(text_list, dir_path, fextList=None):
    dir_files = os.listdir(dir_path)          
    for file in dir_files:        
        file_path = os.path.join(dir_path, file)  
        if os.path.isfile(file_path):
            if fextList is None:
                text_list.append(file_path)
            else:
                f,fext=os.path.splitext(file_path)
                if fext in fextList:
                    text_list.append(file_path)
                
        if os.path.isdir(file_path):
            _list_dir_helper_(text_list,file_path)
    return text_list
    
def genFileListRecursive(inpath, fextList):  
    text_list = []
    _list_dir_helper_(text_list, inpath, fextList)
    return text_list
    
def genPathList(path):
    filenames = os.listdir(path)
    
    list = []
    for filename in filenames:
        fpath = os.path.join(path,filename)
        if not os.path.isfile(fpath):
            list.append(filename)        
    return list
          
def getArgInCMDLine(argv,str,IsValue):
    argc = len(argv)
    for k in range(argc):
        if cmp(argv[k],str) == 0:
            if IsValue == 1:
                if k < argc-1:
                    return argv[k+1]
                else:
                    return None
            elif IsValue == 0:
                return argv[k]
              
def fileparts(inpath):
    strList = os.path.split(inpath)
    folder = strList[0]
    fileName = strList[1]
    strList = os.path.splitext(fileName)
    baseName = strList[0]
    ext = strList[1]
    return (folder,baseName,ext)
    
def splitpath(inpath):
    items = []
    tmp = os.path.split(inpath)
    if len(tmp[1]) > 0:
        items.append(tmp[1])

    tmppath = tmp[0]
    while(1):
        if len(tmppath) == 0:
            break
        tmp = os.path.split(tmppath)
        if len(tmp[1]) > 0:
            items.append(tmp[1])
        else:
            items.append(tmppath)
            break
        tmppath = tmp[0]

    items.reverse()    
    return items    

    
def longestCommonPrefix(strs):  
    res = ""
    for i in range(len(strs[0])):
        for j in range(1, len(strs)):
            if i>=len(strs[j]) or strs[j][i]!=strs[0][i]:
                return res
        res += strs[0][i]    
    return res


        
def fullfile(*items):
    tmp = os.path.join(*items)
    return tmp

def strListToPath(strList):
    tmppath = strList[0]
    for k in range(1, len(strList)):
        tmppath = os.path.join(tmppath, strList[k])
    return tmppath    
    
def mkdir(path):
    os.mkdir(path)
    
def copyfile(src,dest):
    shutil.copy(src,dest)
    
def exist(path):
    return os.path.exists(path)
    
def calIOUPercent(rectA,rectB):
    # rectA = [top-left.x top-left.y width height]
    Ax = rectA[0]
    Ay = rectA[1]
    AW = rectA[2]
    AH = rectA[3]
    
    Bx = rectB[0]
    By = rectB[1]
    BW = rectB[2]
    BH = rectB[3]   
    
    if(Ax > Bx+BW or Bx > Ax+AW):
        return 0
    if(By > Ay+AH or Ay > By+BH):
        return 0
       
    colInt = min(Ax+AW,Bx+BW) - max(Ax,Bx)
    rowInt = min(Ay+AH,By+BH) - max(Ay,By) 
    
    intersection = colInt * rowInt  
    areaA = AW * AH
    areaB = BW * BH
    iou = 1.0*intersection / float(areaA + areaB - intersection)
    return iou

def hexToDec(hexcolor):
    rgb = [
        (hexcolor >> 16) & 0xff,
        (hexcolor >> 8) & 0xff,
        hexcolor & 0xff
        ]
    return rgb


def rgbToHexColorStr(r, g, b):
    color = "#"
    color += str(hex(r)).replace('x','0')[-2:]
    color += str(hex(g)).replace('x','0')[-2:]
    color += str(hex(b)).replace('x','0')[-2:]
    return color


def hexColorStrToRGB(hexcolor):
    __doc__ = """
    convert hex color string to rgb color
    #00FF00 --> [0 255 0]
    """    
    assert(hexcolor[0] == '#' and len(hexcolor) == 7)
    r = int(hexcolor[1:3],16) 
    g = int(hexcolor[3:5],16) 
    b = int(hexcolor[5:7],16) 
    return [r,g,b]

def colorName2RGB(colorName):
    colorName = colorName.lower()
    if colorName == 'red':  
        return (255,0,0)      
    if colorName == 'green':
        return (0,255,0)
    if colorName == 'yellow':
        return (255,255,0)
    if colorName == 'blue':
        return (0,0,255)
    if colorName == 'magenta':
        return (255,0,255)
    if colorName == 'white':
        return (255,255,255)
    if colorName == 'purple':
        return (128,0,128)
    
    return (0,0,0)

def sub2ind(rs,cs,H,W):
    ylist = np.multiply(rs, W)
    ylist = np.add(ylist, cs)
    return ylist.tolist()

def ind2sub(ks, H, W):
    cs = np.mod(ks, W)
    ts = np.subtract(ks,cs)
    rs = np.divide(ts, W)
    return [rs, cs]

def saveObjectToFile(ofile, inobj):
    fid = open(ofile, 'wb')
    tmpstr = pickle.dumps(inobj)
    fid.write(tmpstr)
    fid.close()
    return None

def loadObjectFromFile(infile, className=None):
    try:
        with open(infile,'rb') as fid:
            obj  = pickle.loads(fid.read())
    except:
        print("Error when loading data from %s" % infile)
        if className is not None:
            obj = className()
        else:
            obj = None
    return obj  

def unionOfList(A,B):
    a = copy.deepcopy(A)
    a.extend(B)
    a = np.unique(a)
    a = a.tolist()
    return a

def andOfList(A,B):
    a = copy.deepcopy(A)
    a.extend(B)
    cd = Counter(a)
    nums = list(cd.keys())
    freqs = list(cd.values())
    c = np.where(np.array(freqs) > 1)
    x = np.array(nums)[c]
    return x

def diffOfList(A,B):
    C = []
    for a in A:
        if a not in B:
            C.append(a)
    return C        

######################################################################################

    
def loadFASTAToMem(file):
    fasta = []
    
    content = readAFile(file)    
    _Len = len(content)
    for k in range(_Len):                    
        if content[k][0] == '>':
            name = content[k]            
            seq = ""
            for k2 in range(k+1,_Len):                    
                if content[k2][0] == '>':
                    break
                seq = seq + content[k2]
                
            oneseq = {'ID':"",'SEQ':""}
            oneseq['ID'] = name                
            oneseq['SEQ'] = seq
            fasta.append(oneseq)
    
    return fasta


def fastaToFile(fasta):    
    _Len = len(fasta)
    for i in range(_Len):
        print(fasta[i]['ID'] + "\n" + fasta[i]['SEQ'])


def extractIdentifier(fastaID,format):
    Len = len(fastaID)
    
    posInID = 0
    ID = []
    for k in range( len(format) ): 
        if posInID >= Len:
            break
        if format[k] == '?':
            posInID += 1
            continue
        elif format[k] == 'X':
            ID.append(fastaID[posInID])
            posInID += 1
        else:
            ID.append(format[k])
    _ID = "".join(ID)

    return _ID
    

def readSeqInMyDB(dbFile):
    """ 
    (xyzStart,seqRef,seq3D,seqSub,trueSS) = ReadSeqInMyDB(dbFile)
    """
        
    content = readAFile(dbFile)
    _Len = len(content)
    
    seqRef = seq3D = seqSub = trueSS = ""
    for k in range(_Len):
        if content[k].find(">Reference Sequence") >= 0:
            seqRef = content[k+1]
            break
    if len(seqRef) <= 0:
        print("Reading seqRef from %s error\n" % dbFile)
        return None
        
    for k in range(_Len):            
        if content[k].find(">Real Sequence") >= 0:
            seq3D = content[k+1]
            break
    if len(seq3D) <= 0:
        print("Reading seq3D from %s error\n" % dbFile)
        return None
                        
    for k in range(_Len):                        
        if content[k].find(">True Secondary") >= 0:
            trueSS = content[k+1]                                    
            break
    if len(trueSS) <= 0:
        print("Reading trueSS from %s error\n" % dbFile)
        return None
    
    if len(seqRef) != len(seq3D) or len(seqRef) != len(trueSS):
        print("Sequence error in %s\n" % dbFile)
        return None
        
    for k in range( len(seq3D) ):
        if seq3D[k] != '-':
            break
    xyzStart = k
    
    for k in range(len(seq3D)-1,-1,-1):
        if seq3D[k] != '-':
            break
    xyzEnd = k
    seqSub = seq3D[xyzStart:xyzEnd+1]
        
    return (xyzStart,seqRef,seq3D,seqSub,trueSS)
            
def readCaXYZInMyDB(dbFile):
    content = readAFile(dbFile)
    _Len = len(content)
    
    seqRef = ""
    for k in range(_Len):
        if content[k].find(">Reference Sequence") >= 0:
            seqRef = content[k+1]
            break
    if len(seqRef) <= 0:
        print("Reading seqRef from %s error\n" % dbFile)
        return None
        
    for k in range(_Len):
        if content[k].find(">Ca XYZ") >= 0:
            break            
    startLine = k + 1    
    
    strX = content[startLine+0].split()
    strY = content[startLine+1].split()
    strZ = content[startLine+2].split()
    
    coorX = map(float,strX)
    coorY = map(float,strY)
    coorZ = map(float,strZ)
    
    _Len = len(seqRef)
    if len(coorX) != _Len or len(coorY) != _Len or len(coorZ) != _Len:
        print("Reading CA coordinates from %s error\n" % dbFile)
        return None
    
    xyz = []
    for k in range(_Len):
        xyz.append(coorX[k])
        xyz.append(coorY[k])
        xyz.append(coorZ[k])
    
    return xyz
    
    
def readPDBInfoFile(file):
    """
    ( (year,mon,day),dataType,resolution) = readPDBInfoFile(file)
    read pdb info file
    """
    content = readAFile(file)
    _Lines = len(content)
    
    releaseDate = ()
    dataType = '?'
    resolution = -1
    
    for k in range(_Lines):    
        _Len = len(content[k])
        if content[k].find(">ReleaseTime") >= 0:
            """
            >ReleaseTime: 2006 : 12 : 27
            """
            tmp = content[k].split()
            releaseDate = (int(tmp[1]),int(tmp[3]),int(tmp[5]) )
            break
            
    for k in range(_Lines):    
        if content[k].find(">DataType") >= 0:
            """
            >DataType: X  2.20
            """
            tmp = content[k].split()
            dataType = tmp[1]
            resolution = float(tmp[2])
            break
                        
    return (releaseDate,dataType,resolution)

