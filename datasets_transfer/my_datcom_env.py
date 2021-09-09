import numpy as np
from os import path
import os 
import argparse
ROUND_FACTOR = 4
import subprocess
import re

''' Function to split the output file in all the blocks
that contains the aerodynamic values. It returns a list
with all the blocks'''

def block_split(file_handler):
    pattern = re.compile(r'\*+ \bFLIGHT CONDITIONS AND REFERENCE QUANTITIES \*+')
    linestring = file_handler.read()

    blocks=pattern.split(linestring)
    #Useless first block
    blocks = blocks[1:len(blocks)]
    
    return blocks

''' This small function replace all the bad character in a list of strings
with a _ and a '' (last character) '''

def replaceBadChars(listofStrings):
    list = [];
    pattern = re.compile(r'[./\-]');
    
    for name in listofStrings:
        list.append(pattern.sub('_',name[0:len(name)-1])+
        pattern.sub('',name[len(name)-1]));
        
    return list;

''' This function strip the whitespaces among coefficients name and
returns a list with the name of the coefficients for each block '''

def get_coeffs_name(blocks):
    cnames = []
    pattern = re.compile(r' *\bALPHA\b *([\w -/]*)')


    for block in blocks:
        cnames.append(pattern.findall(block))
    
        cnames_ok = []

        #Loop for stripping middle whitespaces
        for cname in cnames:
            ''' If coefficient are written in two lines i have for single cname a list object. I join the multiple list in one
            then i strip the extra whitespaces'''
            if len(cname)>1:
                dummy=' '.join(cname)
                dummy = dummy.split()
                dummy = replaceBadChars(dummy)
                cnames_ok.append(dummy)
            else:
                for c in cname:
                    dummy = c.split()
                    dummy = replaceBadChars(dummy)
                    cnames_ok.append(dummy)

    return cnames_ok

''' This function is intended to retrieve all the starting values
of each block parsed by block_split()'''

def get_data(blocks):
    # TODO difference between 0.001 and .001
    pattern1 = re.compile(r' *\bMACH NO\b *= * (-*[\d]+.[\d]*)')
    pattern2 = re.compile(r' *\bALTITUDE\b *= * (-*[\d]+.[\d]*)')
    pattern3 = re.compile(r' *\bSIDESLIP\b *= * (-*.[\d]*)') 
    pattern4 = re.compile(r' *\bMOMENT CENTER\b *= * (-*.[\d]*)')


    machs=[]
    alts=[]
    sslips=[]
    mcenter=[]
    
    for block in blocks:
        machs.append(float(pattern1.search(block).group(1)))
        alts.append(float(pattern2.search(block).group(1)))
        sslips.append(float(pattern3.search(block).group(1)))
        mcenter.append(float(pattern4.search(block).group(1)))
    
    return (np.asarray(machs), np.asarray(alts), np.asarray(sslips),
            np.asarray(mcenter))

''' This function loop over the blocks and returns all the data in a 
list
---- OUTPUT 
- raw_data: the output tuple with all the cofficients
- alpha: the AoA values
'''
def get_rawdata(blocks):
    #All the numbers or dashes followed by a number;
    pattern = re.compile('[-.\d](?=[.\d\.])')
    #pattern1 = re.compile('.[\d\]')
    pattern2 = re.compile(r'\n\t?')

    raw_data = []
    for block in blocks:
        # Splitting the block in lines
        lines = pattern2.split(block)
        new_1 = []
        
        #Looping over the lines,
        # if they match the pattern
        # i get the values
        for line in lines:
            #pdb.set_trace()
            line = line.strip()
            
            if pattern.match(line):
                #pdb.set_trace()
                new_2=[]
                service_list = line.split()
                
                for num in service_list:
                    new_2.append(float(num))
                new_1.append(new_2)

   #         elif pattern1.match(line):
   #             new_2=[]
   #             service_list = line.split()
                
    #            for num in service_list:
    #                new_2.append(float(num))
     #           new_1.append(new_2)

                #END FOR SERVICE_LIST
            #ENDIF
        #ENDFOR LINE
    
        # Dividing the alphas value from 
        # the coefficients value;
        alpha = []
        new_1_1 = []
        for row in new_1:
            alpha.append(row[0]) 
            new_1_1.append(row[1:len(new_1[0])]) # length issue
        size = len(new_1_1[0])
        new_1 = new_1_1
    
            
        # Checking if the current block has
        # two tables
    
    
        i=0
        for row in new_1_1:
            if not(len(row)==size):
                break
            i = i+1
           # print(i)

        # If it has "i" less than the length 
        # of the array new_1_1 I have to merge the rows
        # of the first block with the second
        if i<len(new_1_1):
            new_1 = []
            new_2_2=new_1_1[i:len(new_1_1)]
            new_1_1=new_1_1[0:i]
        
            for row1,row2 in zip(new_1_1,new_2_2):
                new_1.append(row1+row2)

                
            #And the alpha array is got twice
            # so it has to be splitted
                alpha = alpha[0:int(len(alpha)/2)];      

     
    
            
    
        #Create Array    
        raw_data.append(np.asanyarray(new_1))

    return np.asarray(alpha),raw_data

''' Main function that parse all the text file and save the data
tables in .mat file '''

def read_data(filename):  
    #Open the file
    fh = open(filename)
    
    #Use the filename for output
    filename = filename.split('.dat')[0]
    
    #Split the blocks 
    blocks = block_split(fh)

    if len(blocks) != 0:
        #Retrieve Coeffs name
        names = get_coeffs_name(blocks)
        
        #Retrieve reference values of Mach NO, Altitude , Sideslip
        [M,A,B,XM] = get_data(blocks)
        
        #Coefficients "matrix" and AoA values
        alpha,data = get_rawdata(blocks)
        
        #print(data)
        #print(alpha)
        #Creating the coefficient list without replications
        realM = []
        realA = []
        realB = []
        realXM = []
        
        realM.append(M[0])
        realA.append(A[0])
        realB.append(B[0])
        realXM.append(XM[0])
        
        for i in range(len(M)):
            if not(M[i] in realM):
                realM.append(M[i])
            if not(A[i] in realA):
                realA.append(A[i])
            if not(B[i] in realB):
                realB.append(B[i])
            if not(XM[i] in realXM):
                realB.append(XM[i])
        
        # Creating State Dictionary with all the state variables
        stateDict = {'Machs':realM,
                    'Alphas':alpha,
                    'Betas':realB,
                    'Altitudes':realA,
                    'Mom. Center':realXM
                    }
            

        #Num. of elements
        lM = len(realM)
        lA = len(realA)
        lB = len(realB)
        lalpha = len(alpha)
        
        
        #Checking how many indipendent blocks i have
        cname = names[0][0]
        j=1

        for i in range(len(blocks)-1):
            new_cname = names[i+1][0]
            
            if new_cname == cname:
                break
            j=j+1
        
        
        #Joining all the coefficients names of the j indipendent blocks
        totalc = []
        for i in range(j):
            totalc = totalc + names[i]

        
        
        #Number of coefficients
        numofc = len(totalc)
        lTrial = len(blocks)//j
        
        final_data = {}
        #I have to generate <numofc> 4D matrixes (preallocation)
        for i in range(numofc):
            final_data[totalc[i]]=np.zeros((lalpha,lM,lB,lA,lTrial))

        # Filling the 4D matrixes with data
        currentBigBlock = np.zeros((lalpha,numofc))

        iM = 0
        iB = 0
        iA = 0
        iTrial = 0
        for i in range(0,len(blocks),j):
            
            #Joining the j indipendent blocks in one
            l=0
            for k in range(i,i+j,1):
                #Sizes
                n,m = data[k].shape
                currentBigBlock[:,l:l+m]=data[k]
                l=l+m
               
            #Populating the final data matrix
            for cname,iC in zip(totalc,range(numofc)):
                
                final_data[cname][:,iM,iB,iA,iTrial] = currentBigBlock[:,iC]
            
            #Checking which parameter has changed
            
            if iM < lM-1:
                iM=iM+1;
            else:
                iM=0;
                if iB < lB-1:
                    iB = iB+1;
                else:
                    iB=0;
                    if iA < lA-1:
                        iA = iA+1;
                    else:
                        iA = 0
                        iTrial += 1;
        mdict={'Coeffs':final_data,'State':stateDict}
    else:
        mdict = {}
    return mdict

class myDatcomEnv():

    def __init__(self, transfer = False):
        self.transfer = transfer
        self.scen = 2
        self.action_size = 5
        self.reward_fail = -5.0
        self.XLE1       = 1.72
        self.XLE2       = 3.2 
        self.CHORD1_1   = 0.29
        self.CHORD1_2   = 0.06
        self.CHORD2_1   = 0.38
        self.CHORD2_2   = 0.19
        self.SSPAN1_2   = 0.23
        self.SSPAN2_2   = 0.22
        self.state = np.round(self._get_obs(), ROUND_FACTOR)

        # Lower and Upper bound 
        # States = [XLE1, XLE2, CHORD1_1, CHORD1_2, CHORD2_1, CHORD2_2, SSPAN1_2, SSPAN2_2]
        self.state_upper = np.array([1.75 , 3.2 , 0.4 , 0.09 , 0.4 , 0.25 , 0.3 , 0.3],
                        dtype=np.float32)

        self.state_lower = np.array([1.25 , 3.0 , 0.1 , 0.0 , 0.1 , 0.0 , 0.1 , 0.1],
                        dtype=np.float32)


    def step(self, parameterization):
        self.XLE1, self.XLE2, self.CHORD1_1, self.CHORD1_2, self.CHORD2_1, self.CHORD2_2, self.SSPAN1_2, self.SSPAN2_2 = \
            parameterization["XLE1"], parameterization["XLE2"], parameterization["CHORD1_1"], parameterization["CHORD1_2"],\
            parameterization["CHORD2_1"], parameterization["CHORD2_2"], parameterization["SSPAN1_2"], parameterization["SSPAN2_2"]

        self.state = np.round(self._get_obs(), ROUND_FACTOR)

        self.print_input('for005.dat')
        if os.path.exists('for006.dat'):
            os.remove('for006.dat')

        
        while not os.path.exists('for006.dat'):
            p = subprocess.run('wine datcom97.exe',shell=True)
        
        """
        parser = argparse.ArgumentParser(description='Parse the DATCOM output file')
        parser.add_argument('filename',nargs='?',help='''The name of the DATCOM 
        output file. Default: "for006.dat"''',default='for006.dat')         

        args = parser.parse_args()
        """
        mdict = read_data("for006.dat")
        
        if bool(mdict):
            
            self.cl_cd = mdict['Coeffs']["CL_CD"][1][0][0][0][0]
            self.xcp = mdict['Coeffs']["X_C_P"][1][0][0][0][0] * -1 * (0.18/3.69)
            self.cd = mdict['Coeffs']["CD"][1][0][0][0][0]
            self.cl = mdict['Coeffs']["CL"][1][0][0][0][0]
            self.costs = self.cl_cd

        else:
            self.costs = self.reward_fail

        print("STEP REWARD : {}".format(self.costs))
        print("CL/CD : {}".format(self.cl_cd))

        return self.cl, self.cd, self.xcp, self.cl_cd

    def reset(self):

        #self.XLE1       = 1.5
        #self.XLE2       = 3.1 
        #self.CHORD1_1   = 0.25
        #self.CHORD1_2   = 0.045
        #self.CHORD2_1   = 0.25
        #self.CHORD2_2   = 0.125
        #self.SSPAN1_2   = 0.20
        #self.SSPAN2_2   = 0.20


        self.XLE1       = 1.72
        self.XLE2       = 3.2 
        self.CHORD1_1   = 0.29
        self.CHORD1_2   = 0.06
        self.CHORD2_1   = 0.38
        self.CHORD2_2   = 0.19
        self.SSPAN1_2   = 0.23
        self.SSPAN2_2   = 0.22

        #self.XLE1       = 1.72
        #self.XLE2       = 3.2 
        #self.CHORD1_1   = 0.29
        #self.CHORD1_2   = 0.06
        #self.CHORD2_1   = 0.38
        #self.CHORD2_2   = 0.25
        #self.SSPAN1_2   = 0.23
        #self.SSPAN2_2   = 0.3

        self.state = np.round(self._get_obs(), ROUND_FACTOR)

        # Initial parameters for the cost
        # Baseline CL/CD and XCP 
        self.print_input('for005.dat')
        if os.path.exists('for006.dat'):
            os.remove('for006.dat')

        while not os.path.exists('for006.dat'):
            s = subprocess.run('wine datcom97.exe',shell=True)
    

        """
        parser = argparse.ArgumentParser(description='Parse the DATCOM output file')
        parser.add_argument('filename',nargs='?',help='''The name of the DATCOM 
        output file. Default: "for006.dat"''',default='for006.dat')         

        args = parser.parse_args()
        """
        mdict = read_data("for006.dat")
        self.base_cl_cd = mdict['Coeffs']["CL_CD"][1][0][0][0][0]
        self.base_xcp = mdict['Coeffs']["X_C_P"][1][0][0][0][0] * -1 * (0.18/3.69)
        self.base_cd = mdict['Coeffs']["CD"][1][0][0][0][0]  
        self.cl_cd = self.base_cl_cd
        self.xcp = self.base_xcp
        self.cd = self.base_cd

        return self.state

    def step_batch(self, batch):
        self.file_name = "for005.dat"

        if os.path.exists(self.file_name):
            os.remove(self.file_name)
        else:
            print("The file does not exist")

        with open(self.file_name, 'a') as fileOut:
            for item1 in batch:
                self.XLE1, self.XLE2, self.CHORD1_1, self.CHORD1_2, self.CHORD2_1, self.CHORD2_2, self.SSPAN1_2, self.SSPAN2_2 = \
                    item1[0], item1[1], item1[2], item1[3], item1[4], item1[5], item1[6], item1[7]
                self.state = np.round(self._get_obs(), ROUND_FACTOR)

                for item in self.intro.keys():
                    print(item,sep=' ',file=fileOut)
                self.write_block(self.FLTCON,fileOut)
                self.write_block(self.REFQ,fileOut)
                self.write_block(self.AXIBOD,fileOut)
                self.write_block(self.FINSET1,fileOut)
                self.write_block(self.FINSET2,fileOut)
                self.write_block(self.DEFLCT,fileOut)
            
                print(" ",file=fileOut ) 
                for item in self.finish.keys():
                    print(item,sep=' ',file=fileOut)
                
        
        if os.path.exists('for006.dat'):
            os.remove('for006.dat')

        while not os.path.exists('for006.dat'):
            s = subprocess.run('wine datcom97.exe',shell=True)

        mdict = read_data("for006.dat")
        cl_cd = mdict['Coeffs']["CL_CD"][1][0][0][0]
        xcp = mdict['Coeffs']["X_C_P"][1][0][0][0] * -1 * (0.18/3.69)
        cd = mdict['Coeffs']["CD"][1][0][0][0]  

        return cl_cd, xcp, cd

    def _get_obs(self):


        self.intro = {"DIM M "       : "",
                "SOSE"          : ""}

        self.FLTCON = {  "name"              : "$FLTCON", 
                    " NMACH="           : [1.0], 
                    " MACH="            : [2.0],
                    " ALT="             : [5000.0],
                    " NALPHA="          : [2.0],
                    " ALPHA="           : [0.0, 4.0],
                    " PHI="             : [0.0],  
                    " end"              : "$"}

        self.REFQ = {    "name"              : "$REFQ",
                    " LREF="            : [0.18],
                    " SREF="            : [0.025447],
                    " XCG="             : [0.0],
                    " end"              : "$"}


        self.AXIBOD = {  "name"              :"$AXIBOD",
                    " TNOSE="           : ["CONE"],
                    " LNOSE="           : [0.49],
                    " DNOSE="           : [0.18],
                    " LCENTR="          : [3.2],
                    " DCENTR="          : [0.18],
                    " end"              : "$"}

        if self.transfer == False and self.scen == 1:
            print("DDPG: Senaryo 1")
            self.FINSET1 = { "name"              : "$FINSET1",
                        " SSPAN="           : [0.09, float(self.SSPAN1_2*0.7)],
                        " CHORD="           : [float(self.CHORD1_1*0.8), float(self.CHORD1_2)],
                        " XLE="             : [float(self.XLE1)],
                        " SWEEP="           : [0.0],
                        " STA="             : [1.0],
                        " NPANEL="          : [4.0],
                        " PHIF="            : [0.0,90.0,180.0,270.0],
                        " ZUPPER="          : [0.025,0.025],
                        " LMAXU ="          : [0.25,0.25],
                        " LFLATU="          : [0.25,0.25],
                        " end"              : "$"}
        elif self.transfer == False and self.scen == 2:
            print("DDPG: Senaryo 2")
            self.FINSET1 = { "name"              : "$FINSET1",
                        " SSPAN="           : [0.09, float(self.SSPAN1_2*0.85)],
                        " CHORD="           : [float(self.CHORD1_1*0.85), float(self.CHORD1_2)],
                        " XLE="             : [float(self.XLE1)],
                        " SWEEP="           : [0.0],
                        " STA="             : [1.0],
                        " NPANEL="          : [4.0],
                        " PHIF="            : [0.0,90.0,180.0,270.0],
                        " ZUPPER="          : [0.025,0.025],
                        " LMAXU ="          : [0.25,0.25],
                        " LFLATU="          : [0.25,0.25],
                        " end"              : "$"}

        else:
            print("Transfer Learning:")
            self.FINSET1 = { "name"              : "$FINSET1",
                        " SSPAN="           : [0.09, float(self.SSPAN1_2)],
                        " CHORD="           : [float(self.CHORD1_1), float(self.CHORD1_2)],
                        " XLE="             : [float(self.XLE1)],
                        " SWEEP="           : [0.0],
                        " STA="             : [1.0],
                        " NPANEL="          : [4.0],
                        " PHIF="            : [0.0,90.0,180.0,270.0],
                        " ZUPPER="          : [0.025,0.025],
                        " LMAXU ="          : [0.25,0.25],
                        " LFLATU="          : [0.25,0.25],
                        " end"              : "$"}


        self.FINSET2 ={  "name"              : "$FINSET2",
                    " SSPAN="           : [0.09, float(self.SSPAN2_2)],
                    " CHORD="           : [float(self.CHORD2_1), float(self.CHORD2_2)],
                    " XLE="             : [float(self.XLE2)],
                    " SWEEP="           : [0.0],
                    " STA="             : [1.0],
                    " NPANEL="          : [4.0],
                    " PHIF="            : [0.0,90.0,180.0,270.0],
                    " ZUPPER="          : [0.025,0.025], 
                    " LMAXU ="          : [0.25,0.25],
                    " LFLATU="          : [0.25,0.25],
                    " end"              : "$"}	

        self.DEFLCT = {  "name"              : "$DEFLCT",
                    " DELTA1="          : [0.0,0.0,0.0,0.0],
                    " DELTA2="          : [0.0,0.0,0.0,0.0],
                    " end"              : "$"}	


        self.finish = {"SAVE"       : "",
                "NEXT CASE"          : ""}

        


        return np.array([self.XLE1, self.XLE2, \
                         self.CHORD1_1, self.CHORD1_2, \
                         self.CHORD2_1, self.CHORD2_2, \
                         self.SSPAN1_2, self.SSPAN2_2])

    def write_block(self,dict,fileout):

        print("\n",file=fileout )        
        for k,item in enumerate(dict.keys()):
            if item == "name":
                print(dict["name"],file=fileout)
            elif k == (len(dict.keys())-2):
                #print(list(dict.keys())[k],*list(dict.values())[k],",","$",sep=' ',file=fileout)
                print(list(dict.keys())[k],','.join(map(str,list(dict.values())[k])),",","$",sep='',file=fileout)
                break    
            else:
                # print(item,*dict[item],",",sep=' ',file=fileout)
                print(item,','.join(map(str,dict[item])),",",sep='',file=fileout)

    def print_input(self,file_name):
        self.file_name = file_name

        if os.path.exists(self.file_name):
            os.remove(self.file_name)
        else:
            print("The file does not exist")

        with open(self.file_name, 'a') as fileOut:
            for item in self.intro.keys():
                print(item,sep=' ',file=fileOut)
            self.write_block(self.FLTCON,fileOut)
            self.write_block(self.REFQ,fileOut)
            self.write_block(self.AXIBOD,fileOut)
            self.write_block(self.FINSET1,fileOut)
            self.write_block(self.FINSET2,fileOut)
            self.write_block(self.DEFLCT,fileOut)
            
            print(" ",file=fileOut ) 
            for item in self.finish.keys():
                print(item,sep=' ',file=fileOut)


