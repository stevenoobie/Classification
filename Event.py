class Event:
    def __init__(self,fLength,fWidth,fSize,fConc,fConc1,fAsym,fM3Long,fM3Trans,fAlpha,fDist,clas):
        self.fLength=fLength
        self.fWidth=fWidth
        self.fSize=fSize
        self.fConc=fConc
        self.fConc1=fConc1
        self.fAsym=fAsym
        self.fM3Long=fM3Long
        self.fM3Trans=fM3Trans
        self.fAlpha=fAlpha
        self.fDist=fDist
        self.clas=clas[0]
    def isGamma(self):
        if self.clas=='g':
            return True
        return False
    def getArray(self):
        arr=[]
        arr.append(self.fLength)
        arr.append(self.fWidth)
        arr.append(self.fSize)
        arr.append(self.fConc)
        arr.append(self.fConc1)
        arr.append(self.fAsym)
        arr.append(self.fM3Long)
        arr.append(self.fM3Trans)
        arr.append(self.fAlpha)
        arr.append(self.fDist)
        return arr